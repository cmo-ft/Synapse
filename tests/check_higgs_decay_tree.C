#include "higgs_decay.hxx"
#include <stdexcept>

void check_higgs_decay_tree() {
  using namespace higgsdecay;
  auto check = [](bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
  };

  // Two jets can swap their truth partons without changing the pair label.
  Inventory inventory;
  inventory.daughters = {{2, 0, 0., 0.}, {2, 0, 0.01, 0.}};
  const std::vector<Object> jets = {{3, 0, 0.004, 0., true}, {4, 0, 0.006, 0., true}};
  auto result = match(inventory, jets, 2);
  check(result.ids[3] == 1 && result.ids[4] == 1, "Both jets belong to H1");
  check((result.known[3] & result.same[3] & (1 << 4)) != 0, "The jets share H1");

  inventory.daughters[1].parent = 4;
  result = match(inventory, jets, both);
  check(result.ancestry[3] == 1 && result.ancestry[4] == 1, "Both jets have Higgs ancestry");
  check(result.ids[3] == -1 && result.ids[4] == -1, "H1 and H2 are interchangeable");
  check((result.known[3] & (1 << 4)) && !(result.same[3] & (1 << 4)), "The jets have different Higgs parents");

  // H -> W+ W-, W+ -> tau+ nu, W- -> e- nu. The stored tree stops at the tau.
  Inventory tau;
  const std::vector<int> ws = {24, -24};
  const std::vector<double> directions = {0., 0.};
  const std::vector<std::vector<int>> children = {{-15, 16}, {11, -12}};
  const std::vector<std::vector<double>> angles = {{0., 0.}, {0., 0.}};
  readHiggs(tau, 2, 25, ws, directions, directions, children, angles, angles);
  const std::vector<Object> leptons = {
    {0, kind(-13), 1., 0., true}, {1, kind(11), 0., 0., true}, {2, kind(13), 2., 0., true}
  };
  result = match(tau, leptons, 2);
  check(result.ancestry[0] == -1, "An unresolved tau+ may produce this mu+");
  check(result.ancestry[1] == 1, "The electron matches the recorded W- daughter");
  check(result.ancestry[2] == 0, "The closed decay record excludes the prompt mu-");
  check(!(result.known[0] & (1 << 1)), "The mu+ and electron pair is unresolved");

  result = match(Inventory{}, leptons, 0);
  check(result.ancestry[0] == 0 && result.ancestry[1] == 0, "An empty Higgs record has no Higgs daughters");
  check((result.known[0] & (1 << 1)) && !(result.same[0] & (1 << 1)), "The pair has no common Higgs");
  check(result.ancestry[9] == -1 && result.known[9] == 0, "MET is excluded");
}
