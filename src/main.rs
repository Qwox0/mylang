use clap::Parser;
use mylang::{
    cli::{Cli, Command},
    compiler::{CompileMode, compile},
    context::CompilationContext,
};

fn main() -> ! {
    let mut cli = Cli::parse();
    let ctx = CompilationContext::new();
    let res = match &mut cli.command {
        Command::Repl {} => todo!("repl"),
        Command::Clean(_) => todo!("clean"),
        Command::Build(args) => compile(ctx.0, CompileMode::Build, args),
        Command::Run(args) => compile(ctx.0, CompileMode::Run, args),
        Command::Check(args) => compile(ctx.0, CompileMode::Check, args),
    };
    std::process::exit(res.exit_code())
}
