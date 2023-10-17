#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

/// How to run
/// 1. Put your input data into points.hex and scalars.hex files in Data folder
/// 2. Parse these values from files using MSMReadHexPoints and MSMReadHexScalars functions
/// 3. Run the msmPreprocessPoints function to get the preprocess points
/// 4. Run the MSMrun function to get the results(you can specify the number of batch operations with the command line argument)
/// 5. You can draw a plot with the given results using build_plot function

mod MSM;
mod host_reduce;
mod host_curve;
mod utility;
mod batch_functions;
mod MSM_cpu;
mod grafics;
mod wgsl_call;

/// functions for parsing points and scalars values from files
pub use MSM::{MSMReadHexPoints, MSMReadHexScalars};
/// functions for converting slices from one type to another
pub use utility::{u32_as_mut_slice_u8, u8_as_mut_slice_u32, u32_as_mut_slice_u64, u8_as_slice_u32};
/// curve functions running on cpu using Rust
pub use host_curve::*;
/// curve functions running on cpu using Rust with the number of batch operations
pub use batch_functions::*;
/// the MSM function running on cpu using Rust
pub use MSM_cpu::*;
/// function for building a plot from given points
pub use grafics::*;
/// functions for calling the WGSL functions from Rust
pub use wgsl_call::*;

/// structure that contains all necessary information for MSM function
pub use crate::MSM::MSMContext;
