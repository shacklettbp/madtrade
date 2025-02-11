#include "mgr.hpp"

using namespace madtrade;
using namespace madrona;

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  Manager mgr(Manager::Config {
    .execMode = ExecMode::CPU,
    .numWorlds = 1,
    .numAgentsPerWorld = 4,
  });

  mgr.init();

  mgr.setAction(0, 0, Action { OrderType::Ask, 50, 4 });
  mgr.setAction(0, 1, Action { OrderType::Ask, 60, 4 });
  mgr.setAction(0, 2, Action { OrderType::Bid, 53, 4 });
  mgr.setAction(0, 3, Action { OrderType::Ask, 55, 4 });

  mgr.step();

  mgr.setAction(0, 0, Action { OrderType::Bid, 52, 4 });
  mgr.setAction(0, 1, Action { OrderType::Ask, 61, 4 });
  mgr.setAction(0, 2, Action { OrderType::Bid, 54, 4 });
  mgr.setAction(0, 3, Action { OrderType::Bid, 54, 4 });

  mgr.step();

  return 0;
}
