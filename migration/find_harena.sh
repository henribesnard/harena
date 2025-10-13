#!/bin/bash
echo "Finding harena directory..."
find /home/ec2-user -name "harena" -type d 2>/dev/null || echo "Not in /home/ec2-user"
find /opt -name "harena" -type d 2>/dev/null || echo "Not in /opt"
echo ""
echo "Checking home directory:"
ls -la /home/ec2-user/
echo ""
echo "Checking /opt:"
ls -la /opt/
echo ""
echo "Looking for .env files:"
find /home -name ".env" 2>/dev/null | head -5
find /opt -name ".env" 2>/dev/null | head -5
