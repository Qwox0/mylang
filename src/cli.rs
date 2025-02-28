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

#[derive(clap::Args, Debug)]
pub struct BuildArgs {
    #[arg(default_value = ".")]
    pub path: PathBuf,

    #[arg(short = 'O', default_value = "0")]
    pub optimization_level: u8,

    #[arg(long = "target")]
    pub target_triple: Option<String>,

    #[arg(long, default_value = "exe")]
    pub out: OutKind,

    #[arg(long)]
    pub no_prelude: bool,

    /// Disabled in benchmarks
    #[clap(skip = true)]
    pub print_compile_time: bool,

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
}

#[derive(clap::ValueEnum, Debug, Clone, PartialEq)]
pub enum OutKind {
    None,

    #[clap(name = "obj")]
    ObjectFile,

    #[clap(name = "exe")]
    Executable,
}

impl BuildArgs {
    /// for benchmarks
    pub fn bench_args() -> Self {
        BuildArgs {
            path: PathBuf::new(),
            optimization_level: 0,
            target_triple: None,
            out: OutKind::None,
            no_prelude: true,
            print_compile_time: false,
            debug_ast: false,
            debug_types: false,
            debug_typed_ast: false,
            debug_functions: false,
            debug_llvm_ir_unoptimized: false,
            debug_llvm_ir_optimized: false,
        }
    }
}
