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
        python main.py replay <task_id>
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
        "current_year": current_year
    }


# ---------------------------
# Run modes
# ---------------------------
def run():
    """Run the crew for a customer support case."""
    inputs = get_inputs()
    try:
        SupportSystem().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    inputs = get_inputs()
    try:
        SupportSystem().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        SupportSystem().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and return results."""
    inputs = get_inputs()
    try:
        SupportSystem().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
