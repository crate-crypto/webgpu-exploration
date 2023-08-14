struct chain_t {
  firstOperation: bool,
  // value that is stored in registers
  value: u32,
}

// chain_t impl

fn initialize_chain_t1(self_: ptr<function, chain_t>) { // +
  (*self_).firstOperation=true;
}

fn initialize_chain_t2(self_: ptr<function, chain_t>, carry: bool) { //+
  (*self_).firstOperation=false;
  if carry {
    (*self_).value = uadd_cc(1u, 0xFFFFFFFFu);
  }
  else {
    (*self_).value = uadd_cc(0u, 0xFFFFFFFFu);
  }
}

fn reset1(self_: ptr<function, chain_t>) { // +
  (*self_).firstOperation=true;
  (*self_).value = uadd_cc(0u, 0u);
}

fn reset2(self_: ptr<function, chain_t>, carry: bool) { // +
  (*self_).firstOperation=false;
  if carry {
    (*self_).value = uadd_cc(1u, 0xFFFFFFFFu);
  }
  else {
    (*self_).value = uadd_cc(0u, 0xFFFFFFFFu);
  }
}

fn getCarry(self_: ptr<function, chain_t>) -> bool { // +
  return uadd_cc(0u, 0u)!=0u;
}

fn add_chain_t(self_: ptr<function, chain_t>, a: u32, b: u32) -> u32 { // +
  if (*self_).firstOperation {
    (*self_).value = uadd_cc(0u, 0u);
  }
  (*self_).firstOperation = false;
  return uadd_cc(a, b);
}

fn sub_chain_t(self_: ptr<function, chain_t>, a: u32, b: u32) -> u32 { // +
  if (*self_).firstOperation {
    (*self_).value = uadd_cc(1u, 0xFFFFFFFFu);
  }
  (*self_).firstOperation = false;
  return usubc_cc(a, b);
}

fn madwide1(self_: ptr<function, chain_t>, a: u32, b: u32, c: vec2<u32>) -> vec2<u32> { // +
  if (*self_).firstOperation {
    (*self_).value = uadd_cc(0u, 0u);
  }
  (*self_).firstOperation = false;
  return u2madwidec_cc(a, b, c);
}

fn madwide2(self_: ptr<function, chain_t>, a: u32, b: u32, c: WideNumber) -> WideNumber { // +
  if (*self_).firstOperation {
    (*self_).value = uadd_cc(0u, 0u);
  }
  (*self_).firstOperation = false;
  return madwidec_cc(a, b, c);
}
  