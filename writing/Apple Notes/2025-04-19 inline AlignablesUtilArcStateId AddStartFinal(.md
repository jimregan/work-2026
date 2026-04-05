inline AlignablesUtil::Arc::StateId AddStartFinal(
    VectorFst<AlignablesUtil::Arc> *fst) {
  auto start = fst->AddState();
  fst->SetStart(start);
  fst->SetFinal(start, AlignablesUtil::Arc::Weight::One());
  return start;
}

self.pair_fsa = fst.VectorFst()
start_state = self.pair_fsa.add_state()
self.pair_fsa.set_start(start_state)
self.pair_fsa.set_final(start_state, fst.Weight.one(self.pair_fsa.weight_type()))