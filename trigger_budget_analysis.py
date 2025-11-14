#!/usr/bin/env python3
"""
Script to manually trigger budget profile analysis for a user
"""
import sys
import requests
from datetime import timedelta

# Add the user_service to the path to import create_access_token
sys.path.insert(0, '/app/user_service')

from user_service.core.security import create_access_token

def trigger_budget_analysis(user_id: int):
    """Trigger budget profile analysis for a user"""

    # Generate JWT for the user
    user_jwt = create_access_token(
        subject=user_id,
        permissions=["chat:write"],
        expires_delta=timedelta(minutes=30)
    )

    print(f"Generated JWT for user {user_id}")

    # Call budget profiling service
    budget_url = "http://harena_budget_profiling_service:3006/api/v1/budget/profile/analyze"
    headers = {
        "Authorization": f"Bearer {user_jwt}",
        "Content-Type": "application/json"
    }

    print(f"Calling budget profiling service at {budget_url}")

    try:
        response = requests.post(budget_url, headers=headers, timeout=180)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Budget profile created successfully!")
            print(f"User segment: {result.get('user_segment')}")
            print(f"Average monthly income: {result.get('avg_monthly_income')}")
            print(f"Average monthly expenses: {result.get('avg_monthly_expenses')}")
            print(f"Savings rate: {result.get('savings_rate')}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trigger_budget_analysis.py <user_id>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    trigger_budget_analysis(user_id)
