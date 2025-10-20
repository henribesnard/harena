"""
Script de test rapide pour conversation_service_v3
"""
import asyncio
import httpx


async def test_service():
    """Teste le service conversation_service_v3"""

    base_url = "http://localhost:3008"

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🚀 Testing Conversation Service v3...\n")

        # Test 1: Health check
        print("1️⃣ Health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            return

        # Test 2: Root endpoint
        print("2️⃣ Root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"   Status: {response.status_code}")
            data = response.json()
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Architecture: {data.get('architecture')}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

        # Test 3: Agent health check
        print("3️⃣ Agent health check...")
        try:
            response = await client.get(f"{base_url}/api/v3/conversation/health")
            print(f"   Status: {response.status_code}")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Search service: {data.get('search_service')}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

        # Test 4: Simple question
        print("4️⃣ Testing conversation...")
        try:
            response = await client.post(
                f"{base_url}/api/v3/conversation/ask",
                json={
                    "user_id": 1,
                    "message": "Combien j'ai dépensé en courses ce mois-ci ?",
                    "conversation_id": "test_001"
                }
            )
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   Success: {data.get('success')}")
                print(f"   Message preview: {data.get('message', '')[:150]}...")
                print(f"   Total results: {data.get('total_results')}")
                print(f"   Metadata: {data.get('metadata', {})}\n")
            else:
                print(f"   Error: {response.text}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

        # Test 5: Stats
        print("5️⃣ Getting stats...")
        try:
            response = await client.get(f"{base_url}/api/v3/conversation/stats")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                orch_stats = data.get('orchestrator', {})
                print(f"   Total queries: {orch_stats.get('total_queries')}")
                print(f"   Success rate: {orch_stats.get('success_rate', 0):.2%}")
                print(f"   Corrections needed: {orch_stats.get('corrections_needed')}")
                print("\n✅ All tests completed!")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Conversation Service v3 - Test Script")
    print("=" * 60 + "\n")

    asyncio.run(test_service())
