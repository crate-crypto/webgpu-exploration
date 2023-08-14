use rand::Rng;

use crate::{host_curve::*, MSMContext, msmPreprocessPoints_cpu, MSM_run_cpu, GpuConsts};

pub fn setZero_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    setZero(&mut array);
  }
}

pub fn setOne_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    setOne(&mut array);
  }
}

pub fn setR_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    setR(&mut array);
  }
}

pub fn set_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let field = [1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    set(&mut array, field);
  }
}

pub fn load_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let field = [1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    load(&mut array, &field);
  }
}

pub fn store_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let field = [1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    store(&mut array, field);
  }
}

pub fn isZero_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    isZero(array);
  }
}

pub fn addN_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let field = [1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    addN(&mut array, field);
  }
}

pub fn add_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let a = [1,2,3,4,5,6,7,8,9,10,11,12];
  let b = [12,11,10,9,8,7,6,5,4,3,2,1];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    add(&mut array, a, b);
  }
}

pub fn sub_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let a = [1,2,3,4,5,6,7,8,9,10,11,12];
  let b = [12,11,10,9,8,7,6,5,4,3,2,1];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    sub(&mut array, a, b);
  }
}

pub fn mul_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let a = [1,2,3,4,5,6,7,8,9,10,11,12];
  let b = [12,11,10,9,8,7,6,5,4,3,2,1];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    mul(&mut array, a, b);
  }
}

pub fn swap_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let mut a = [1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    swap(&mut array, &mut a);
  }
}

pub fn reduce_batch() {
  let mut rng = rand::thread_rng();
  let mut array = [0; 12];
  let a = [1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      array[j] = rng.gen_range(1..=100);
    }

    reduce(&mut array, a);
  }
}

pub fn reduce_PointXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let mut p: PointXYZZ = PointXYZZ{x, y, zz, zzz};

    p.reduce();
  }
}

pub fn load_PointXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  let a = [1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let mut p: PointXYZZ = PointXYZZ{x, y, zz, zzz};

    p.load(&a);
  }
}

pub fn store_PointXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  let mut a = [1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12];

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let p: PointXYZZ = PointXYZZ{x, y, zz, zzz};

    p.store(&mut a);
  }
}

pub fn normalize_PointXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let mut p: PointXYZZ = PointXYZZ{x, y, zz, zzz};

    p.normalize();
  }
}

pub fn setZero_AccumulatorXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let p: PointXYZZ = PointXYZZ{x, y, zz, zzz};
    let mut acc: AccumulatorXYZZ = AccumulatorXYZZ{xyzz: p };

    acc.setZero();
  }
}

pub fn dbl_AccumulatorXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  let p1: PointXYZZ = PointXYZZ::default();

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let p: PointXYZZ = PointXYZZ{x, y, zz, zzz};
    let mut acc: AccumulatorXYZZ = AccumulatorXYZZ{xyzz: p };

    acc.dbl(p1);
  }
}

pub fn add_AccumulatorXYZZ_batch() {
  let mut rng = rand::thread_rng();
  let mut x = [0; 12];
  let mut y = [0; 12];
  let mut zz = [0; 12];
  let mut zzz = [0; 12];

  let p1: PointXYZZ = PointXYZZ::default();

  for _ in 0..10000 {
    for j in 0..12 {
      x[j] = rng.gen_range(1..=100);
      y[j] = rng.gen_range(1..=100);
      zz[j] = rng.gen_range(1..=100);
      zzz[j] = rng.gen_range(1..=100);
    }

    let p: PointXYZZ = PointXYZZ{x, y, zz, zzz};
    let mut acc: AccumulatorXYZZ = AccumulatorXYZZ{xyzz: p };

    acc.add(p1);
  }
}

pub fn msmPreprocessPoints_cpu_batch() {
  let mut rng = rand::thread_rng();
  let mut affinePointsPtr: [u32; 48] = [0; 48];
  let mut msm: MSMContext = MSMContext::new(100000, 100000);

  for _ in 0..100000 {
    for j in 0..48 {
      affinePointsPtr[j] = rng.gen_range(1..=100);
    }

    msmPreprocessPoints_cpu(&mut msm, &affinePointsPtr, 65536);
  }
}

pub fn msmRun_cpu_batch() {
  let mut rng = rand::thread_rng();
  let mut projectiveResultsPtr = [0; 12];
  let mut scalarsPtr = [0; 12];
  let mut scalars: u32;
  let mut msm: MSMContext = MSMContext::new(100000, 100000);

  for _ in 0..10000 {
    for j in 0..12 {
      projectiveResultsPtr[j] = rng.gen_range(1..=100);
      scalarsPtr[j] = rng.gen_range(1..=100);
    }
    scalars = rng.gen_range(1..=100);

    MSM_run_cpu(&mut msm, &mut projectiveResultsPtr, &scalarsPtr, scalars);
  }
}

pub fn msmPreprocessPoints_gpu_batch() {
  let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11];
  let mut msm: MSMContext = MSMContext::new(100000, 100000);
  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  for _ in 0..1000000 {
    msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu);
  }
}