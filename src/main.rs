use clap::Parser;
use mylang::{
    cli::{Cli, Command},
    compiler::{CompileMode, compile},
};

fn main() -> ! {
    let cli = Cli::parse();
    let res = match cli.command {
        Command::Repl {} => todo!("repl"),
        Command::Clean(_) => todo!("clean"),
        Command::Build(args) => compile(CompileMode::Build, args),
        Command::Run(args) => compile(CompileMode::Run, args),
        Command::Check(args) => compile(CompileMode::Check, args),
    };
    std::process::exit(res.exit_code())
}
