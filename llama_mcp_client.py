import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import websockets
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LlamaMCPClient")

class LlamaMCPClient:
    """
    MCP Client that integrates with LLAMA.CPP to enable function calling
    with MCP Server tools.
    """
    
    def __init__(self,
        model_path: str,
        mcp_server_uri: str = "ws://localhost:9000",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        """
        Initialize the LLAMA.CPP MCP Client
        
        Args:
            model_path: Path to GGUF model file
            mcp_server_uri: WebSocket URI of MCP server
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            verbose: Enable verbose logging
        """
        self.mcp_server_uri = mcp_server_uri
        self.model_path = model_path
        
        # Initialize LLAMA.CPP
        logger.info(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )
        logger.info("Model loaded successfully")
        
        # Define available tools (from MCP Server)
        self.tools = self._define_tools()
        
    def _define_tools(self) -> List[Dict[str, Any]]:
        """
        Define tools available from MCP Server in OpenAI function calling format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image using ComfyUI based on a text prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The text prompt describing the image to generate"
                            },
                            "width": {
                                "type": "integer",
                                "description": "Image width in pixels",
                                "default": 512
                            },
                            "height": {
                                "type": "integer",
                                "description": "Image height in pixels",
                                "default": 512
                            },
                            "workflow_id": {
                                "type": "string",
                                "description": "ComfyUI workflow ID to use",
                                "default": "basic_api_test"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model checkpoint filename",
                                "default": "v1-5-pruned-emaonly.safetensors"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]
    
    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP Server via WebSocket
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        payload = {
            "tool": tool_name,
            "params": json.dumps(params)
        }
        
        try:
            async with websockets.connect(self.mcp_server_uri) as ws:
                logger.info(f"Calling MCP tool: {tool_name} with params: {params}")
                await ws.send(json.dumps(payload))
                response = await ws.recv()
                result = json.loads(response)
                logger.info(f"Tool result: {result}")
                return result
        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            return {"error": str(e)}
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Chat with the LLM with function calling support
        
        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming (not fully supported yet)
            
        Returns:
            Final response from the LLM
        """
        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            system_message = {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to image generation tools. "
                    "When users ask you to create or generate images, use the generate_image function. "
                    "Always provide clear and helpful responses."
                )
            }
            messages = [system_message] + messages
        
        # Chat loop with function calling
        while True:
            # Generate response with tools
            response = self.llm.create_chat_completion(
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            assistant_message = response["choices"][0]["message"]
            messages.append(assistant_message)
            
            # Check if LLM wants to call a function
            tool_calls = assistant_message.get("tool_calls")
            
            if not tool_calls:
                # No function call, return final response
                return assistant_message.get("content", "")
            
            # Process each tool call
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                function_name = function.get("name")
                function_args = json.loads(function.get("arguments", "{}"))
                
                logger.info(f"LLM requested tool call: {function_name}")
                
                # Execute the tool via MCP Server
                tool_result = asyncio.run(
                    self.call_mcp_tool(function_name, function_args)
                )
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(tool_result)
                })
            
            # Continue loop to let LLM process tool results
    
    def simple_chat(self, user_message: str) -> str:
        """
        Simple chat interface with a single user message
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        messages = [
            {"role": "user", "content": user_message}
        ]
        return self.chat(messages)


def main():
    """
    Example usage of LlamaMCPClient
    """
    # Initialize client
    # NOTE: Replace with your actual model path
    MODEL_PATH = "./models/llama-3.1-8b-instruct-q4_k_m.gguf"
    
    try:
        client = LlamaMCPClient(
            model_path=MODEL_PATH,
            mcp_server_uri="ws://localhost:9000",
            n_ctx=4096,
            n_gpu_layers=-1  # Use GPU if available
        )
        
        print("\n" + "="*60)
        print("LLAMA.CPP MCP Client Initialized")
        print("="*60 + "\n")
        
        # Interactive chat loop
        print("Type your messages (or 'quit' to exit):\n")
        
        chat_history = []
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to history
            chat_history.append({"role": "user", "content": user_input})
            
            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = client.chat(chat_history.copy())
            print(response + "\n")
            
            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": response})
            
    except FileNotFoundError:
        print(f"\nError: Model file not found at {MODEL_PATH}")
        print("Please download a GGUF model and update MODEL_PATH")
        print("\nRecommended models:")
        print("- Llama 3.1 8B Instruct (supports function calling)")
        print("- Download from: https://huggingface.co/models?search=llama-3.1+gguf")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()