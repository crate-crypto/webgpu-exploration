use fp::run;

fn main() {
    env_logger::init();
    pollster::block_on(run(&vec![1, 2, 4, 5], "main"));
}
