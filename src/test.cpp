#include "mgr.hpp"

using namespace madtrade;
using namespace madrona;

void testNPCSystem() {
  printf("\n=== Testing NPC System ===\n");

  // Initialize simulator with 2 agents and 2 NPCs
  Manager mgr(Manager::Config {
    .execMode = ExecMode::CPU,
    .numWorlds = 1,
    .numAgentsPerWorld = 1,
    .numNPCsPerWorld = 2,
    .D = 10,  // Secret numbers will be between 0 and 9
    .simFlags = SimFlags::InterpretAddAsReplace,
    .settlementPrice = 100,
    .gpuID = 0,
  });
  
  mgr.init();

  // Test 1: Check NPC initialization
  printf("\nTest 1: NPC Order Generation\n");
  mgr.setAction(0, 0, Action { OrderType::Hold, 0, 0 });
  mgr.step();  // This will trigger initialization prints

  // Test 2: Check order matching with NPCs
  printf("\nTest 3: Order Matching with NPCs\n");
  
  mgr.setAction(0, 0, Action { OrderType::Bid, 5, 5 });

  // Step the simulation
  mgr.step();

  printf("NPC System Test Complete!\n");
}

void testBasicTrading() {
  printf("\n=== Testing Basic Trading ===\n");

  // Create manager with single world containing 4 agents
  Manager mgr(Manager::Config {
    .execMode = ExecMode::CPU,
    .numWorlds = 1,
    .numAgentsPerWorld = 4,
    .numNPCsPerWorld = 0,
    .simFlags = SimFlags::InterpretAddAsReplace,
    .gpuID = 0,
  });

  // Initialize simulation
  mgr.init();

  printf("\nTest 1: Single Bid\n");
  mgr.setAction(0, 0, Action { OrderType::Bid, 100, 1});
  mgr.setAction(0, 1, Action { OrderType::Hold, 0, 0});
  mgr.setAction(0, 2, Action { OrderType::Hold, 0, 0});
  mgr.setAction(0, 3, Action { OrderType::Hold, 0, 0});

  mgr.step();

  printf("\nTest 2: Second Bid\n");
  mgr.setAction(0, 0, Action { OrderType::Hold, 0, 0});
  mgr.setAction(0, 1, Action { OrderType::Bid, 90, 1});
  mgr.setAction(0, 2, Action { OrderType::Hold, 0, 0});
  mgr.setAction(0, 3, Action { OrderType::Hold, 0, 0});

  mgr.step();

  printf("\nTest 3: Ask Order\n");
  mgr.setAction(0, 0, Action { OrderType::Hold, 0, 0});
  mgr.setAction(0, 1, Action { OrderType::Hold, 0, 0});
  mgr.setAction(0, 2, Action { OrderType::Ask, 80, 1});
  mgr.setAction(0, 3, Action { OrderType::Hold, 0, 0});

  mgr.step();

  printf("Basic Trading Test Passed!\n");
}

int main(int argc, char *argv[])
{
  // Unused command line args
  (void)argc;
  (void)argv;

  testNPCSystem();
  testBasicTrading();

  return 0;
}
