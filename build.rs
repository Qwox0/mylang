use std::path::Path;

fn symlink(original: &str, link: impl AsRef<Path>) {
    let link = link.as_ref();
    println!("link '{}' -> '{original}", link.display());

    if let Some(link_parent) = link.parent() {
        std::fs::create_dir_all(link_parent).unwrap()
    }

    #[cfg(not(unix))]
    panic!("Sorry, no symlinks for you.");

    #[cfg(unix)]
    {
        if std::fs::symlink_metadata(link).is_ok_and(|m| m.is_symlink()) {
            std::fs::remove_file(link).unwrap()
        }
        std::os::unix::fs::symlink(original, link).unwrap()
    }
}

fn main() {
    for mode in ["debug", "release"] {
        symlink("../../lib", &format!("./target/{mode}/lib"));
        symlink("../../../lib", &format!("./target/{mode}/deps/lib"));
    }
    /*
    symlink("../../lib", "./target/debug/lib");
    symlink("../../../lib", "./target/debug/deps/lib");

    symlink("../../lib", "./target/release/lib");
    symlink("../../../lib", "./target/release/deps/lib");
    */
}
