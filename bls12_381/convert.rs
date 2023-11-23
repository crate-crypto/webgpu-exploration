#! /usr/bin/env rust-script 

use std::io::read_to_string;

fn main() {

    let mystr = read_to_string(std::io::stdin()).unwrap();

    // let mystr = r#"
    //             0x3e2f585da55c9ad1,
    //             0x4294213d86c18183,
    //             0x382844c88b623732,
    //             0x92ad2afd19103e18,
    //             0x1d794e4fac7cf0b9,
    //             0x0bd592fc7d825ec8,
    //     "#; 
    
    println!("str is {}", mystr);
    let my_val = mystr.split(',').
        map(|val| String::from(val.trim())).
        filter(|val| {
            val.len() != 0
        }).map(|val| format!("\n\t0x{}u,\n\t{}u,",&val[10..],&val[0..10]))
        .chain([String::from(")")])
        .fold(String::from("array<u32,12>("), |acc, item| format!("{acc}{item}"));


    println!("{}", my_val);
}
