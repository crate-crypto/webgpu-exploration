#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use crate::host_curve::PointXYZZ;
use crate::host_curve::AccumulatorXYZZ;
use crate::host_curve::exportField;

#[derive(Default)]
pub struct HostReduce {
  pub windows: u32,
  pub windowBits: u32,
  pub warps: u32,
  pub warpsPerWindow: u32,
  pub bucketsPerThread: u32,
}

impl HostReduce {
  pub fn initialize(&mut self, initialWindows: u32, initialWindowBits: u32, initialWarps: u32) {
    self.windows=initialWindows;
    self.windowBits=initialWindowBits;
    self.warps=initialWarps;

    self.warpsPerWindow=self.warps/self.windows;
    self.bucketsPerThread=((1<<self.windowBits-1)+self.warpsPerWindow*32-1)/(self.warpsPerWindow*32);
  }

  fn reduceWindow(&mut self, result: &mut AccumulatorXYZZ, warpResults: &[u32]) {
    let mut sum: AccumulatorXYZZ = AccumulatorXYZZ::default();
    let mut sos: AccumulatorXYZZ = AccumulatorXYZZ::default();
    let mut interior: AccumulatorXYZZ = AccumulatorXYZZ::default();

    let mut point: PointXYZZ = PointXYZZ::default();
    let mut scaleAmount: i32 = self.bucketsPerThread as i32;

    result.setZero();

    for i in (0..=self.warpsPerWindow-1).rev() {
      if i*3*48 + 96 + 48 > warpResults.len() as u32 {
        point.load(&warpResults);
        result.add(point);
        point.load(&warpResults); 
        interior.add(point);
        point.load(&warpResults);
      }
      else {
        point.load(&warpResults[(i*3*48 + 0) as usize ..]);
        result.add(point);
        point.load(&warpResults[(i*3*48 + 96) as usize ..]); 
        interior.add(point);
        point.load(&warpResults[(i*3*48 + 48) as usize ..]);
      }

      if i>0 {
        sum.add(point);
        sos.add(sum.xyzz);
      }
    }

    for _ in 0..5 {
      sos.dbl(sos.xyzz);
    }
    interior.add(sos.xyzz);

    while scaleAmount!=0 {
      if (scaleAmount & 0x01)!=0 {
        result.add(interior.xyzz);
      }
      interior.dbl(interior.xyzz);
      scaleAmount=scaleAmount>>1;
    }
  }

  pub fn reduce(&mut self, msmResult: &mut [u64], warpResults: &[u32]) {
    let mut result: AccumulatorXYZZ = AccumulatorXYZZ::default();
    let mut windowResult: AccumulatorXYZZ = AccumulatorXYZZ::default();

    for _ in (0..=self.windows-1).rev() {
      for _ in 0..self.windowBits {
        result.dbl(result.xyzz);
      }
      self.reduceWindow(&mut windowResult, &warpResults);
      result.add(windowResult.xyzz);
    }
    result.xyzz.normalize();

    // result.xyzz.dump();   // useful for debugging

    if msmResult.len() < 18 {
      return;
    }

    exportField(msmResult, result.xyzz.x);
    exportField(&mut msmResult[6..], result.xyzz.y);
    exportField(&mut msmResult[12..], result.xyzz.zz);
  }
}