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
    /// Opens a REPL
    Check(BuildArgs),
    Clean(BuildArgs),

    Repl {},

    Dev {},
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

    #[arg(long, value_enum)]
    pub debug_tokens: bool,
    #[arg(long)]
    pub debug_ast: bool,
    #[arg(long)]
    pub debug_types: bool,
    #[arg(long)]
    pub debug_typed_ast: bool,
    #[arg(long)]
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
