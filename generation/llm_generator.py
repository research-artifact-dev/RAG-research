#!/usr/bin/env python3
# generation/llm_generator.py
"""
LLM Generator using SAP AI Core Generative AI Hub.

Calls GPT-5.3 CODEX model deployed in SAP AI Core to generate code candidates.
"""

import sys
import os
from typing import List, Dict
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, NUM_CANDIDATES

# Load environment variables from .env file if it exists
load_dotenv()

class LLMGenerator:
    """
    LLM Generator for code generation using SAP AI Core.
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        """
        Initialize LLM generator with SAP AI Core connection.

        Args:
            model_name: Model name in SAP AI Core (e.g., 'gpt-4', 'gpt-35-turbo')
        """
        self.model_name = model_name
        self._setup_connection()

    def _setup_connection(self):
        """
        Setup connection to SAP AI Core Generative AI Hub.

        Supports two methods:
        Method 1: Individual environment variables
        - AICORE_AUTH_URL
        - AICORE_CLIENT_ID
        - AICORE_CLIENT_SECRET
        - AICORE_BASE_URL
        - AICORE_RESOURCE_GROUP

        Method 2: Complete service key JSON
        - AICORE_SERVICE_KEY (entire JSON as string)
        """
        try:
            # Check if we have service key JSON
            service_key_json = os.getenv('AICORE_SERVICE_KEY')

            if service_key_json:
                # Method 2: Load from JSON
                import json
                print("Loading SAP AI Core credentials from AICORE_SERVICE_KEY...")

                service_key = json.loads(service_key_json)

                # Set individual environment variables from JSON
                os.environ['AICORE_CLIENT_ID'] = service_key['clientid']
                os.environ['AICORE_CLIENT_SECRET'] = service_key['clientsecret']
                os.environ['AICORE_AUTH_URL'] = service_key['url']
                os.environ['AICORE_BASE_URL'] = service_key['serviceurls']['AI_API_URL'] + '/v2'

                # Resource group (use default if not in JSON)
                if 'AICORE_RESOURCE_GROUP' not in os.environ:
                    os.environ['AICORE_RESOURCE_GROUP'] = 'default'

                print("✓ Credentials loaded from service key JSON")
            else:
                # Method 1: Individual variables already set
                print("Using individual AICORE_* environment variables...")

            # Import appropriate SDK based on model type
            if "anthropic" in self.model_name or "claude" in self.model_name.lower():
                # Use Bedrock Session for Anthropic models
                from gen_ai_hub.proxy.native.amazon.clients import Session
                self.session = Session()
                self.api_type = "anthropic"
                print(f"✓ Using Bedrock/Anthropic API")
            elif "gemini" in self.model_name.lower():
                # Use Client for Gemini models via SAP AI Core
                from gen_ai_hub.proxy.native.google_genai.clients import Client
                from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
                self.proxy_client = get_proxy_client('gen-ai-hub')
                self.google_client = Client(proxy_client=self.proxy_client)
                self.api_type = "gemini"
                print(f"✓ Using Google Gemini API")
            else:
                # Use OpenAI API for GPT models
                from gen_ai_hub.proxy.native.openai import chat
                self.chat = chat
                self.api_type = "openai"
                print(f"✓ Using OpenAI API")

            print(f"✓ Connected to SAP AI Core")
            print(f"✓ Model: {self.model_name}")

        except ImportError:
            print("✗ Error: SAP AI SDK not installed")
            print("Install: pip install 'sap-ai-sdk-gen[all]'")
            print("(Old package 'gen-ai-hub-sdk' is archived)")
            raise
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing AICORE_SERVICE_KEY JSON: {e}")
            print("Make sure the JSON is properly formatted")
            raise
        except KeyError as e:
            print(f"✗ Error: Missing key in service key JSON: {e}")
            print("Required keys: clientid, clientsecret, url, serviceurls.AI_API_URL")
            raise
        except Exception as e:
            print(f"✗ Error connecting to SAP AI Core: {e}")
            print("\nMethod 1: Set individual variables:")
            print("  - AICORE_AUTH_URL")
            print("  - AICORE_CLIENT_ID")
            print("  - AICORE_CLIENT_SECRET")
            print("  - AICORE_BASE_URL")
            print("  - AICORE_RESOURCE_GROUP")
            print("\nMethod 2: Set complete JSON:")
            print("  - AICORE_SERVICE_KEY (entire service key JSON)")
            raise

    def generate_candidates(
        self,
        messages: List[Dict],
        n: int = NUM_CANDIDATES,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS
    ) -> List[str]:
        """
        Generate multiple code candidates using SAP AI Core LLM.

        Args:
            messages: Chat messages in OpenAI format
            n: Number of candidates to generate
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens per candidate

        Returns:
            List of generated code strings
        """
        candidates = []

        print(f"\n🤖 Generating {n} code candidates...")
        print(f"   Model: {self.model_name}")
        print(f"   Temperature: {temperature}")

        try:
            # Generate candidates one by one
            for i in range(n):
                print(f"   Generating candidate {i+1}/{n}...")

                if self.api_type == "anthropic":
                    # Use Bedrock/Anthropic API
                    bedrock = self.session.client(model_name=self.model_name)

                    # Convert OpenAI format to Anthropic format
                    # Extract user message (last message with role='user')
                    user_content = None
                    for msg in reversed(messages):
                        if msg["role"] == "user":
                            user_content = msg["content"]
                            break

                    # Extract system message if exists
                    system_content = None
                    for msg in messages:
                        if msg["role"] == "system":
                            system_content = msg["content"]
                            break

                    # Build Anthropic message format
                    anthropic_message = {
                        "role": "user",
                        "content": [{"text": user_content}]
                    }

                    # Call converse API
                    response = bedrock.converse(
                        messages=[anthropic_message],
                        inferenceConfig={
                            "maxTokens": max_tokens,
                            "temperature": temperature
                        },
                        system=[{"text": system_content}] if system_content else None
                    )

                    code = response["output"]["message"]["content"][0]["text"].strip()

                elif self.api_type == "gemini":
                    # Use Google Gemini API
                    # Convert OpenAI format to Gemini format
                    gemini_content = []

                    for msg in messages:
                        if msg["role"] == "system":
                            # Prepend system message to first user message
                            continue
                        elif msg["role"] == "user":
                            # Add system message if exists as part of user message
                            text_content = msg["content"]
                            if messages[0].get("role") == "system":
                                text_content = messages[0]["content"] + "\n\n" + text_content

                            gemini_content.append({
                                "role": "user",
                                "parts": [{"text": text_content}]
                            })

                    # Call generate_content via GoogleClient
                    response = self.google_client.models.generate_content(
                        model=self.model_name,
                        contents=gemini_content,
                        config={
                            "max_output_tokens": max_tokens,
                            "temperature": temperature
                        }
                    )

                    code = response.text.strip()

                else:
                    # Use OpenAI API
                    response = self.chat.completions.create(
                        model_name=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    code = response.choices[0].message.content.strip()

                candidates.append(code)

            print(f"✓ Generated {len(candidates)} candidates")
            return candidates

        except Exception as e:
            print(f"✗ Error generating candidates: {e}")
            raise

    def generate_single(
        self,
        messages: List[Dict],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS
    ) -> str:
        """
        Generate a single code candidate.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated code string
        """
        candidates = self.generate_candidates(
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return candidates[0] if candidates else ""


if __name__ == "__main__":
    # Test LLM generator
    print("=" * 70)
    print("LLM GENERATOR TEST")
    print("=" * 70)

    try:
        # Initialize generator
        generator = LLMGenerator()

        # Test prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert Python programmer."
            },
            {
                "role": "user",
                "content": "Write a Python function to sort a dictionary by its values."
            }
        ]

        # Generate candidates
        candidates = generator.generate_candidates(messages, n=2)

        print(f"\nGenerated {len(candidates)} candidates:")
        for i, code in enumerate(candidates, 1):
            print(f"\n{'='*70}")
            print(f"Candidate {i}:")
            print(f"{'='*70}")
            print(code)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nMake sure you have:")
        print("1. Installed: pip install 'gen-ai-hub-sdk'")
        print("2. Set SAP AI Core environment variables")
        print("3. Access to GPT-5.3 CODEX model")
