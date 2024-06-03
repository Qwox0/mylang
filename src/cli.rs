use clap::{CommandFactory, Parser};
use std::path::PathBuf;

#[derive(Debug)]
pub struct Cli {
    pub command: Command,
    pub debug: Option<DebugOptions>,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(args_conflicts_with_subcommands = true)]
struct CliImpl {
    #[command(flatten)]
    default_args: Option<RunScriptArgs>,

    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, value_enum)]
    pub debug: Option<DebugOptions>,
}

// all identical:
// `mylang build.mylang`
// `mylang run-script build.mylang`
// `mylang build`
// `mylang build build.mylang`
#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Interpret a Script or Build script (default)
    RunScript(RunScriptArgs),
    /// Interpret a Build Script to build the current project
    Build {
        /// Build script
        #[arg(default_value = "build.mylang")]
        build_script: PathBuf,
    },
    /// Compile a source file
    Compile {
        file: PathBuf,
    },
    /// Open a REPL
    Repl {},
    Check {},
    Clean {},
}

#[derive(clap::Args, Debug)]
pub struct RunScriptArgs {
    /// Script file to Interpret
    pub script: PathBuf,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugOptions {
    Tokens,
    Ast,
    LlvmIrUnoptimized,
    LlvmIrOptimized,
}

impl Cli {
    pub fn parse() -> Self {
        let parsed = CliImpl::parse();

        let command = parsed
            .command
            .or_else(|| parsed.default_args.map(Command::RunScript))
            .expect("either a command or the default command was parsed");

        Cli { command, debug: parsed.debug }
    }

    pub fn print_help() -> Result<(), std::io::Error> {
        CliImpl::command().print_help()
    }
}
