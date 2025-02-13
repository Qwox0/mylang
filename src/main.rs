use clap::Parser;
use mylang::{
    cli::{Cli, Command},
    compiler::{CompileMode, compile2},
};

fn main() -> ! {
    let cli = Cli::parse();
    let exit_code = match &cli.command {
        Command::Repl {} => todo!("repl"),
        Command::Clean(_) => todo!("clean"),
        Command::Build(args) => compile2(CompileMode::Build, args),
        Command::Run(args) => compile2(CompileMode::Run, args),
        Command::Check(args) => compile2(CompileMode::Check, args),
    };
    std::process::exit(exit_code)
}
