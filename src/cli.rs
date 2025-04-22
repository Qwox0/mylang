use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Builds the current project
    Build(BuildArgs),
    /// Builds and runs the current project
    Run(BuildArgs),
    Check(BuildArgs),
    Clean(BuildArgs),

    /// Opens a REPL
    Repl {},
}

#[derive(clap::Args, Debug, Clone)]
pub struct BuildArgs {
    #[arg(default_value = ".")]
    pub path: PathBuf,

    #[arg(short = 'O', default_value = "0")]
    pub optimization_level: u8,

    #[arg(long = "target")]
    pub target_triple: Option<String>,

    #[arg(long, default_value = "exe")]
    pub out: OutKind,

    /// Disabled in benchmarks and tests
    #[clap(skip = true)]
    pub print_compile_time: bool,

    #[arg(long = "lib")]
    pub is_lib: bool,

    /// The name of the first function called by the program
    #[arg(long, default_value = "main")]
    pub entry_point: String,

    #[arg(long)]
    pub debug_ast: bool,
    #[arg(long)]
    pub debug_types: bool,
    #[arg(long)]
    pub debug_typed_ast: bool,
    #[arg(long)]
    pub debug_functions: bool,
    #[arg(long, alias = "debug-llvm-ir")]
    pub debug_llvm_ir_unoptimized: bool,
    #[arg(long)]
    pub debug_llvm_ir_optimized: bool,
    #[arg(long)]
    pub debug_linker_args: bool,
}

impl Default for BuildArgs {
    fn default() -> Self {
        let Command::Build(args) = Cli::parse_from(["", "build"]).command else {
            unreachable!()
        };
        args
    }
}

#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Default)]
pub enum OutKind {
    None,

    #[clap(name = "obj")]
    ObjectFile,

    #[clap(name = "exe")]
    #[default]
    Executable,
}

impl BuildArgs {
    /// for benchmarks
    pub fn comp_bench_args() -> Self {
        Self::test_args(TestArgsOptions {
            debug_ast: false,
            debug_types: false,
            debug_typed_ast: false,
            llvm_optimization_level: 0,
            print_llvm_module: false,
            entry_point: "main",
        })
    }

    /// for tests
    pub fn test_args(opt: TestArgsOptions) -> Self {
        BuildArgs {
            path: PathBuf::from("test.mylang"),
            optimization_level: opt.llvm_optimization_level,
            target_triple: None,
            out: OutKind::None,
            print_compile_time: false,
            is_lib: false,
            entry_point: opt.entry_point.to_string(),
            debug_ast: opt.debug_ast,
            debug_types: opt.debug_types,
            debug_typed_ast: opt.debug_typed_ast,
            debug_functions: false,
            debug_llvm_ir_unoptimized: false,
            debug_llvm_ir_optimized: opt.print_llvm_module,
            debug_linker_args: false,
        }
    }
}

#[derive(Debug)]
pub struct TestArgsOptions {
    pub debug_ast: bool,
    pub debug_types: bool,
    pub debug_typed_ast: bool,
    pub llvm_optimization_level: u8,
    pub print_llvm_module: bool,
    pub entry_point: &'static str,
}

impl Default for TestArgsOptions {
    fn default() -> Self {
        Self {
            debug_ast: Default::default(),
            debug_types: Default::default(),
            debug_typed_ast: Default::default(),
            llvm_optimization_level: Default::default(),
            print_llvm_module: Default::default(),
            entry_point: "test",
        }
    }
}
