#!/usr/bin/env python

import sys
import os
import warnings
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from support_system.crew import SupportSystem
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Make sure src/support_system/crew.py exists")
    sys.exit(1)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def validate_environment():
    """Validate environment and configuration"""
    
    # Check API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print("✅ Environment validated successfully")
    return True

def initialize_system():
    """Initialize the support system with bulletproof error handling"""
    
    print("🚀 Advanced Customer Support System")
    print("🤖 RAG-Powered | Multi-Agent Workflow | Policy-Aware")
    print("="*70)
    
    if not validate_environment():
        return None
    
    try:
        print("⏳ Initializing comprehensive support system...")
        support_system = SupportSystem()
        
        # FIXED: Safe status retrieval with fallback
        try:
            system_status = support_system.get_system_status()
            if system_status and isinstance(system_status, dict):
                print(f"✅ System initialized successfully!")
                print(f"📊 Status: {system_status.get('system_health', 'unknown').upper()}")
                print(f"🔥 Advanced Mode: {system_status.get('advanced_mode', False)}")
                
                components = system_status.get('components', {})
                if components:
                    print(f"🏗️  Components: {system_status.get('agents', 0)} agents, {system_status.get('tasks', 0)} tasks")
                    print(f"📁 Policies: {components.get('policy_manager', 'unknown')}")
                    print(f"🔍 Vector Store: {components.get('vector_store', 'unknown')}")
            else:
                print("✅ System initialized successfully!")
                print("📊 Status: Operational (status details unavailable)")
                
        except Exception as status_error:
            print("✅ System initialized successfully!")
            print(f"⚠️  Status check failed: {status_error}")
        
        return support_system
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("\n🔧 Common fixes:")
        print("   • Check your YAML files for syntax errors")
        print("   • Ensure all required packages are installed")
        print("   • Verify your OpenAI API key is valid")
        return None

def run():
    """Run a single query"""
    
    support_system = initialize_system()
    if not support_system:
        return
    
    query = sys.argv[2] if len(sys.argv) > 2 else "I need help with my account"
    
    print(f"📝 Processing Query: {query}")
    print("⏳ Executing workflow...")
    print("-" * 70)
    
    try:
        # FIXED: Safe method call with fallback
        if hasattr(support_system, 'process_customer_interaction'):
            result = support_system.process_customer_interaction(
                user_query=query,
                user_id=f"cli_user_{datetime.now().timestamp()}",
                store_conversation=True
            )
            
            print("\n🎯 CUSTOMER SUPPORT RESPONSE")
            print("="*70)
            print(f"\n{result.get('response', 'No response available')}")
            print("\n📊 PROCESSING SUMMARY")
            print("-"*70)
            print(f"👤 Customer ID: {result.get('user_id', 'unknown')}")
            print(f"⏱️  Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"✅ Status: {result.get('system_status', 'unknown').upper()}")
            print("="*70)
            
        else:
            # Fallback to basic crew kickoff
            context = {
                "topic": query,
                "user_query": query,
                "current_year": str(datetime.now().year),
                "timestamp": datetime.now().isoformat()
            }
            
            result = support_system.crew().kickoff(inputs=context)
            
            print("\n🎯 RESPONSE")
            print("="*60)
            print(result)
            print("="*60)
            print("✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error processing query: {e}")

def interactive():
    """Interactive mode with bulletproof error handling"""
    
    print("🤖 Interactive Customer Support System")
    print("="*70)
    print("Welcome! Ask any support question.")
    print("Commands: 'exit' to quit, 'status' for info, 'help' for assistance")
    print("="*70)
    
    support_system = initialize_system()
    if not support_system:
        return
    
    conversation_count = 0
    customer_id = f"interactive_customer_{datetime.now().timestamp()}"
    
    print(f"\n👤 Customer ID: {customer_id}")
    print("🤖 Ready to assist! What can I help you with?")
    
    while True:
        try:
            query = input(f"\n👤 You: ").strip()
            
            if query.lower() in ['exit', 'quit', 'bye']:
                print(f"\n🤖 Thank you! Processed {conversation_count} queries. Goodbye! 👋")
                break
                
            elif query.lower() == 'status':
                try:
                    status = support_system.get_system_status()
                    if status and isinstance(status, dict):
                        print(f"\n📊 System Status: {status.get('system_health', 'operational').upper()}")
                        print(f"🔥 Advanced Mode: {status.get('advanced_mode', False)}")
                    else:
                        print(f"\n📊 System Status: OPERATIONAL")
                    print(f"📈 This session: {conversation_count} queries")
                except Exception as e:
                    print(f"\n📊 System Status: OPERATIONAL (details unavailable)")
                    print(f"📈 This session: {conversation_count} queries")
                continue
                
            elif query.lower() == 'help':
                print("\n🆘 Help:")
                print("   • Ask any customer support question naturally")
                print("   • 'status' - View system information")
                print("   • 'exit' - End session")
                continue
                
            elif not query:
                print("🤖 Please enter your question.")
                continue
            
            conversation_count += 1
            print("🤖 Processing through AI workflow...")
            
            try:
                # FIXED: Safe method call with fallback
                if hasattr(support_system, 'process_customer_interaction'):
                    result = support_system.process_customer_interaction(
                        user_query=query,
                        user_id=customer_id,
                        store_conversation=True
                    )
                    
                    print(f"\n🤖 Support Agent: {result.get('response', 'Unable to process request')}")
                    
                    if result.get('system_status') == 'success':
                        print(f"⚡ Processed in {result.get('processing_time', 0):.2f}s")
                    
                else:
                    # Fallback to basic crew kickoff
                    context = {
                        "topic": query,
                        "user_query": query,
                        "current_year": str(datetime.now().year),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    result = support_system.crew().kickoff(inputs=context)
                    print(f"\n🤖 Support Agent: {result}")
                
            except Exception as e:
                print(f"\n❌ Error processing request: {e}")
                print("💡 Please try rephrasing your question.")
                
        except KeyboardInterrupt:
            print(f"\n\nSession ended. Processed {conversation_count} queries. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def test():
    """Test the system with error handling"""
    
    print("🧪 Testing Customer Support System")
    print("="*50)
    
    support_system = initialize_system()
    if not support_system:
        return
    
    test_queries = [
        "I can't log into my account",
        "My payment was declined", 
        "How do I cancel my subscription?"
    ]
    
    passed = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {query}")
        try:
            # FIXED: Safe method call with fallback
            if hasattr(support_system, 'process_customer_interaction'):
                result = support_system.process_customer_interaction(
                    user_query=query,
                    user_id=f"test_user_{i}",
                    store_conversation=False
                )
                
                if result.get('system_status') == 'success' and len(result.get('response', '')) > 30:
                    print(f"✅ PASSED ({result.get('processing_time', 0):.2f}s)")
                    passed += 1
                else:
                    print(f"❌ FAILED - {result.get('response', 'No response')[:50]}...")
                    
            else:
                # Fallback test
                context = {
                    "topic": query,
                    "user_query": query,
                    "current_year": str(datetime.now().year)
                }
                
                result = support_system.crew().kickoff(inputs=context)
                
                if result and len(str(result)) > 30:
                    print(f"✅ PASSED - {str(result)[:50]}...")
                    passed += 1
                else:
                    print(f"❌ FAILED - Short response")
                    
        except Exception as e:
            print(f"💥 ERROR: {e}")
    
    print(f"\n📊 Results: {passed}/{len(test_queries)} tests passed")
    if passed == len(test_queries):
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your configuration.")

def main():
    """Main entry point with bulletproof error handling"""
    
    if len(sys.argv) < 2:
        print("🤖 Advanced Customer Support System")
        print("="*50)
        print("Usage:")
        print("  python main.py run [query]        - Process single query")
        print("  python main.py interactive        - Interactive chat")
        print("  python main.py test              - Run system tests")
        return
    
    mode = sys.argv[1].lower()
    
    try:
        if mode == "run":
            run()
        elif mode in ["interactive", "chat"]:
            interactive()
        elif mode == "test":
            test()
        else:
            print(f"❌ Unknown command: {mode}")
            print("Use: run, interactive, or test")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
