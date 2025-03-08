#include "mgr.hpp"

using namespace madtrade;
using namespace madrona;

int main(int argc, char *argv[])
{
  // Unused command line args
  (void)argc;
  (void)argv;

  // Create manager with single world containing 4 agents
  Manager mgr(Manager::Config {
    .execMode = ExecMode::CPU,
    .numWorlds = 1,
    .numAgentsPerWorld = 4,
    .simFlags = SimFlags::InterpretAddAsReplace,
  });

  // Initialize simulation
  mgr.init();

  // First round of actions:
  // Agent 0: Ask 4 units at price 50
  mgr.setAction(0, 0, Action { OrderType::Ask, 50, 4 });
  // Agent 1: Ask 4 units at price 60  
  mgr.setAction(0, 1, Action { OrderType::Ask, 60, 4 });
  // Agent 2: Bid 4 units at price 53
  mgr.setAction(0, 2, Action { OrderType::Bid, 53, 4 });
  // Agent 3: Ask 4 units at price 55
  mgr.setAction(0, 3, Action { OrderType::Ask, 55, 4 });

  mgr.step();
  
  printf("\n------ End of Step 1 ------\n");

  // Second round of actions:
  // Agent 0: Bid 4 units at price 52
  mgr.setAction(0, 0, Action { OrderType::Bid, 52, 4 });
  // Agent 1: Ask 4 units at price 61
  mgr.setAction(0, 1, Action { OrderType::Ask, 61, 4 });
  // Agent 2: Bid 4 units at price 54
  mgr.setAction(0, 2, Action { OrderType::Bid, 54, 4 });
  // Agent 3: Bid 4 units at price 54
  mgr.setAction(0, 3, Action { OrderType::Bid, 54, 4 });

  mgr.step();

  printf("\n------ End of Step 2 ------\n");

  return 0;
}
