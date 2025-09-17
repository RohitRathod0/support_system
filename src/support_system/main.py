#!/usr/bin/env python

import sys
import warnings
from datetime import datetime
from support_system.crew import SupportSystem

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ---------------------------
# Utility: Prepare dynamic inputs
# ---------------------------

def get_inputs():
    """
    Get dynamic inputs for the support system.
    Customer topic is passed from command line, otherwise defaults.

    Example usage:
        python main.py run "Payment Issue"
        python main.py train 5 output.json "Order Tracking"
        python main.py replay task_id
        python main.py test 3 gpt-4 "Refund Request"
    """
    # Default topic for testing
    topic = "General Customer Support"
    current_year = str(datetime.now().year)

    # If user provides a topic, use it (last argument is treated as topic)
    if len(sys.argv) > 2 and not sys.argv[-1].isdigit():
        topic = sys.argv[-1]

    return {
        "topic": topic,
        "current_year": current_year,
        "user_id": f"user_{datetime.now().timestamp()}",
        "session_id": f"session_{datetime.now().timestamp()}",
        "timestamp": datetime.now().isoformat(),
        "user_context": "{}",  # Empty JSON string
        "internal_policies": "[]",  # Empty JSON array
        "user_query": topic
    }

# ---------------------------
# Run modes
# ---------------------------

def run():
    """Run the crew for a customer support case."""
    inputs = get_inputs()

    try:
        support_system = SupportSystem()

        # Use the enhanced process_customer_query method if available
        if hasattr(support_system, 'process_customer_query'):
            result = support_system.process_customer_query(
                user_query=inputs["topic"],
                user_id=inputs["user_id"],
                session_id=inputs["session_id"]
            )
            print("\n=== SUPPORT SYSTEM RESULT ===")
            print(f"Response: {result['response']}")
            print(f"User ID: {result['user_id']}")
            print(f"Session ID: {result['session_id']}")
            if 'sources_used' in result:
                print(f"Sources Used: {result['sources_used']}")
            return result
        else:
            # Fallback to standard crew kickoff
            result = support_system.crew().kickoff(inputs=inputs)
            print("\n=== SUPPORT SYSTEM RESULT ===")
            print(result)
            return result

    except Exception as e:
        print(f"An error occurred while running the crew: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")

def train():
    """Train the crew for a given number of iterations."""
    if len(sys.argv) < 3:
        print("Usage: python main.py train <n_iterations> <filename> [topic]")
        print("Example: python main.py train 5 training_results.json")
        return

    inputs = get_inputs()

    try:
        support_system = SupportSystem()
        support_system.crew().train(
            n_iterations=int(sys.argv[2]),
            filename=sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].endswith('.json') else "training_output.json",
            inputs=inputs
        )
        print(f"Training completed with {sys.argv[2]} iterations")

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """Replay the crew execution from a specific task."""
    if len(sys.argv) < 3:
        print("Usage: python main.py replay <task_id>")
        print("Example: python main.py replay task_12345")
        return

    try:
        support_system = SupportSystem()
        support_system.crew().replay(task_id=sys.argv[2])
        print(f"Replay completed for task: {sys.argv[2]}")

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """Test the crew execution and return results."""
    if len(sys.argv) < 4:
        print("Usage: python main.py test <n_iterations> <eval_llm> [topic]")
        print("Example: python main.py test 3 gpt-4 'Billing issue'")
        return

    inputs = get_inputs()

    try:
        support_system = SupportSystem()
        support_system.crew().test(
            n_iterations=int(sys.argv[2]),
            eval_llm=sys.argv[3],
            inputs=inputs
        )
        print(f"Testing completed with {sys.argv[2]} iterations using {sys.argv[3]}")

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def interactive():
    """Start an interactive session with the support system."""
    print("🤖 SupportSystem Interactive Mode")
    print("=" * 50)
    print("Type your customer support queries. Type 'quit' to exit.")
    print("=" * 50)

    support_system = SupportSystem()
    user_id = f"interactive_user_{datetime.now().timestamp()}"

    while True:
        try:
            user_input = input("\n👤 Customer: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 Thank you for using SupportSystem! Goodbye!")
                break

            if not user_input:
                print("🤖 Please enter your question or concern.")
                continue

            print("\n🤖 Processing your request...")

            # Use enhanced functionality if available
            if hasattr(support_system, 'process_customer_query'):
                result = support_system.process_customer_query(
                    user_query=user_input,
                    user_id=user_id,
                    session_id=f"interactive_{datetime.now().timestamp()}"
                )
                print(f"\n🤖 SupportSystem: {result['response']}")
            else:
                # Fallback to standard approach
                inputs = {
                    "topic": user_input,
                    "current_year": str(datetime.now().year),
                    "user_id": user_id,
                    "session_id": f"interactive_{datetime.now().timestamp()}",
                    "timestamp": datetime.now().isoformat(),
                    "user_context": "{}",
                    "internal_policies": "[]",
                    "user_query": user_input
                }
                result = support_system.crew().kickoff(inputs=inputs)
                print(f"\n🤖 SupportSystem: {result}")

        except KeyboardInterrupt:
            print("\n\n🤖 Session ended. Thank you!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try rephrasing your question.")

def show_help():
    """Show help information."""
    print("""
🤖 SupportSystem - AI-Powered Customer Support System
===================================================

Usage Modes:
  python main.py run [query]         - Process a single support query
  python main.py interactive         - Start interactive chat mode
  python main.py train <n> <file>    - Train the system
  python main.py test <n> <model>    - Test system performance
  python main.py replay <task_id>    - Replay a specific task

Examples:
  python main.py run "I need help with billing"
  python main.py interactive
  python main.py train 5 training_results.json
  python main.py test 3 gpt-4 "Password reset issue"

Features:
  ✓ Dynamic customer support responses
  ✓ Company policy compliance (CSV integration)
  ✓ User session management and history
  ✓ External information integration
  ✓ Vector database storage
  ✓ Multi-agent workflow with quality assurance

Environment Setup:
  Required: OPENAI_API_KEY in .env file
  Optional: SEARCH_API_KEY for enhanced web search
    """)

# ---------------------------
# Main execution
# ---------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    mode = sys.argv[1].lower()

    try:
        if mode == "run":
            run()
        elif mode == "interactive" or mode == "chat":
            interactive()
        elif mode == "train":
            train()
        elif mode == "replay":
            replay()
        elif mode == "test":
            test()
        elif mode == "help" or mode == "--help":
            show_help()
        else:
            print(f"Unknown mode: {mode}")
            show_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
