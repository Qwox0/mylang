use crate::{
    arena_allocator::Arena,
    ast::{Ast, debug::DebugAst},
    cli::{BuildArgs, OutKind},
    codegen::llvm,
    parser::{
        StmtIter,
        lexer::{Code, Lexer},
        parser_helper::ParserInterface,
    },
    ptr::Ptr,
    sema::{self, Sema, SemaResult},
    util::{self, display_spanned_error},
};
use inkwell::context::Context;
use std::{
    assert_matches::debug_assert_matches,
    time::{Duration, Instant},
};

#[thread_local]
static mut CODE: Option<Ptr<Code>> = None;

pub fn set_code(c: &Code) {
    unsafe { CODE = Some(Ptr::from_ref(c)) }
}

pub fn code() -> &'static Code {
    #[allow(static_mut_refs)]
    unsafe {
        CODE.as_ref().unwrap()
    }
}

impl<'c, 'alloc> Sema<'c, 'alloc> {
    /// This doesn't panic if an error is found.
    /// the caller has to check if `sema.errors` contains an error
    pub fn analyze_all<'ctx>(&mut self, stmts: &mut [Ptr<Ast>]) -> Vec<usize> {
        for s in stmts.iter().copied() {
            self.preload_top_level(s);
        }

        let mut finished = vec![false; stmts.len()];
        let mut remaining_count = stmts.len();
        let mut order = Vec::with_capacity(stmts.len());
        while finished.iter().any(std::ops::Not::not) {
            let old_remaining_count = remaining_count;
            debug_assert!(stmts.len() == finished.len());
            remaining_count = 0;
            for (idx, (s, finished)) in stmts.iter().zip(finished.iter_mut()).enumerate() {
                if *finished {
                    continue;
                }
                let res = self.analyze_top_level(*s);
                *finished = res != SemaResult::NotFinished;
                match res {
                    SemaResult::Ok(_) => order.push(idx),
                    SemaResult::NotFinished => remaining_count += 1,
                    SemaResult::Err(_) => {},
                }
            }
            // println!("finished statements: {:?}", finished);
            if remaining_count == old_remaining_count {
                /*
                eprintln!("cycle detected");
                for (s, finished) in stmts.iter().zip(finished) {
                    if finished {
                        continue;
                    }
                    eprint!("{:?} -> ", s);
                    let decl = s.try_downcast::<crate::ast::Decl>();
                    eprintln!(
                        "{:?}",
                        decl.map(|d| d.ident.text.as_ref()).unwrap_or("{top level expr}"),
                    );
                }
                */
                /*
                for s in not_finished {
                    println!("{:#?}", s.get::<crate::ast::Decl>().init.unwrap().get::<crate::ast::Fn>());
                }
                */
                panic!("cycle detected") // TODO: find location of cycle
            }
        }

        order
    }
}

impl<'ctx> llvm::Codegen<'ctx, '_> {
    pub fn compile_all(&mut self, stmts: &[Ptr<Ast>], order: &[usize]) {
        debug_assert_eq!(stmts.len(), order.len());
        for idx in order {
            let s = stmts[*idx];
            self.compile_top_level(s);
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum CompileMode {
    Check,
    Build,
    Run,

    // for benchmarks:
    Parse,
    Codegen,
}

pub fn compile2(mode: CompileMode, args: &BuildArgs) -> i32 {
    let mut code = String::with_capacity(4096);
    if !args.no_prelude {
        let prelude_path = concat!(std::env!("HOME"), "/src/mylang/lib/prelude.mylang");
        util::write_file_to_string(prelude_path, &mut code).unwrap();
    }

    if args.path.is_dir() {
        println!("Compiling project at {:?}", args.path);
        for file in args
            .path
            .read_dir()
            .unwrap()
            .map(|f| f.unwrap().path())
            .filter(|p| p.is_file())
            .filter(|p| p.extension().is_some_and(|ext| ext == "mylang"))
        {
            util::write_file_to_string(&file, &mut code).unwrap();
        }
    } else if args.path.is_file() {
        println!("Compiling file at {:?}", args.path);
        util::write_file_to_string(&args.path, &mut code).unwrap();
    } else {
        panic!("{:?} is not a dir nor a file", args.path)
    }
    compile(code.as_ref(), mode, args)
}

#[inline]
pub fn parse(code: &Code, alloc: &Arena) -> Vec<Ptr<Ast>> {
    StmtIter::parse_all_or_fail(code, alloc)
}

pub fn compile(code: &Code, mode: CompileMode, args: &BuildArgs) -> i32 {
    set_code(code);
    let mut compile_time = CompileDurations::default();
    let alloc = Arena::new();

    if args.debug_tokens {
        println!("### Tokens:");
        let mut lex = Lexer::new(code);
        while let Some(t) = lex.next() {
            println!("{:?}", t)
        }
        println!();
    }

    if args.debug_ast {
        println!("### AST Nodes:");
        if let Err(()) = StmtIter::parse_and_debug(code) {
            return 1;
        }
        println!();
    }

    let frontend_parse_start = Instant::now();
    let mut stmts = parse(code, &alloc);
    compile_time.parser = frontend_parse_start.elapsed();

    if mode == CompileMode::Parse {
        return 0;
    }

    // ##### Sema #####

    let sema_start = Instant::now();
    let mut sema = sema::Sema::new(code, &alloc, args.debug_types);
    let order = sema.analyze_all(&mut stmts);
    compile_time.sema = sema_start.elapsed();

    if !sema.errors.is_empty() {
        eprintln!("Sema Error in {:?}", compile_time.sema);
        for e in sema.errors.iter() {
            display_spanned_error(e, sema.code);
        }
        return 1;
    }

    if args.debug_typed_ast {
        println!("\n### Typed AST Nodes:");
        for s in stmts.iter().copied() {
            println!("stmt @ {:x?}", s);
            s.print_tree();
        }
        println!();
    }

    if mode == CompileMode::Check {
        #[cfg(not(test))]
        compile_time.print();
        // println!("{} KiB", alloc.0.allocated_bytes() as f64 / 1024.0);
        return 0;
    }

    // ##### Codegen #####

    let codegen_start = Instant::now();
    let context = Context::create();
    let mut codegen = llvm::Codegen::new_module(&context, "dev", &sema.primitives);
    codegen.compile_all(&stmts, &order);
    compile_time.codegen = codegen_start.elapsed();

    if args.debug_functions {
        print!("functions:");
        for a in codegen.module.0.get_functions() {
            print!("{:?},", a.get_name());
        }
        println!("\n");
    }

    if mode == CompileMode::Codegen {
        return 0;
    }

    // ##### Backend #####

    debug_assert_matches!(mode, CompileMode::Build | CompileMode::Run);

    let backend_setup_start = Instant::now();
    let target_machine = llvm::Codegen::init_target_machine(args.target_triple.as_deref());
    let module = codegen.module;
    compile_time.backend_setup = backend_setup_start.elapsed();

    if args.debug_llvm_ir_unoptimized {
        println!("### Unoptimized LLVM IR:");
        module.0.print_to_stderr();
        println!();
    }

    let backend_start = Instant::now();
    module.optimize(&target_machine, args.optimization_level).unwrap();
    compile_time.optimization = backend_start.elapsed();

    if args.debug_llvm_ir_optimized {
        println!("### Optimized LLVM IR:");
        module.0.print_to_stderr();
        println!();
    }

    if args.path.is_dir() {
        todo!()
    }

    if args.out == OutKind::None {
        #[cfg(not(test))]
        compile_time.print();
        return 0;
    }

    let mut exe_file_path = args.path.with_file_name("out");
    exe_file_path.push(args.path.file_stem().unwrap());
    let obj_file_path = exe_file_path.with_added_extension("o");
    let write_obj_file_start = Instant::now();
    module.compile_to_obj_file(&target_machine, &obj_file_path).unwrap();
    compile_time.writing_obj = write_obj_file_start.elapsed();

    if args.out == OutKind::Executable {
        let linking_start = std::time::Instant::now();
        let err = std::process::Command::new("gcc")
            .arg(obj_file_path.as_os_str())
            .arg("-o")
            .arg(exe_file_path.as_os_str())
            .arg("-lm")
            .status()
            .unwrap();
        if !err.success() {
            println!("linking with gcc failed: {:?}", err);
        }

        compile_time.linking = linking_start.elapsed();
    }

    compile_time.print();

    if args.out == OutKind::Executable && mode == CompileMode::Run {
        println!("\nRunning `{}`", exe_file_path.display());
        if let Err(e) = std::process::Command::new(exe_file_path).status().unwrap().exit_ok() {
            println!("{e}");
            return e.code().unwrap_or(1);
        }
    }
    0
}

#[derive(Default)]
pub struct CompileDurations {
    // frontend:
    parser: Duration,
    sema: Duration,
    codegen: Duration,

    // backend:
    backend_setup: Duration,
    optimization: Duration,
    writing_obj: Duration,

    linking: Duration,
}

impl CompileDurations {
    pub fn print(&self) {
        let frontend_total = self.parser + self.sema + self.codegen;
        println!("  Frontend:              {:?}", frontend_total);
        println!("    Lexer, Parser:         {:?}", self.parser);
        println!("    Semantic Analysis:     {:?}", self.sema);
        if self.codegen.is_zero() {
            return;
        }
        println!("    LLVM IR Codegen:       {:?}", self.codegen);

        let backend_total = self.backend_setup + self.optimization + self.writing_obj;
        println!("  Backend:               {:?}", backend_total);
        println!("    LLVM Setup:            {:?}", self.backend_setup);
        println!("    LLVM pass pipeline:    {:?}", self.optimization);
        if self.writing_obj.is_zero() {
            return;
        }
        println!("    writing obj file:      {:?}", self.writing_obj);

        if self.linking.is_zero() {
            return;
        }
        println!("  Linking with gcc:      {:?}", self.linking);

        let total = frontend_total + backend_total + self.linking;
        println!("  Total:                 {:?}", total);
    }
}
