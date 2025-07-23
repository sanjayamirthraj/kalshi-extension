#!/bin/bash

# Kalshi Markets Cache Refresh Script
# This script calls the refresh endpoint and logs the response

LOG_FILE="/tmp/kalshi_refresh.log"
API_URL="http://localhost:8000/refresh-cache"

echo "$(date): Starting market refresh..." >> "$LOG_FILE"

# Call the refresh endpoint with timeout
response=$(curl -s -w "HTTP_STATUS:%{http_code}" -m 30 -X POST "$API_URL" 2>&1)

# Extract HTTP status and response body
http_status=$(echo "$response" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
response_body=$(echo "$response" | sed 's/HTTP_STATUS:[0-9]*$//')

if [ "$http_status" = "200" ]; then
    echo "$(date): Market refresh successful - $response_body" >> "$LOG_FILE"
else
    echo "$(date): Market refresh failed with status $http_status - $response_body" >> "$LOG_FILE"
fi

# Keep only last 100 lines of log to prevent it from growing too large
tail -n 100 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE" 

# */2 * * * * /path/to/refresh_markets.sh