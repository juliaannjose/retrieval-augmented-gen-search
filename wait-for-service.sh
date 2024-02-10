#!/bin/sh

# Function to wait for service availability
wait_for_service() {
    local url=$1
    local retries=${2:-120}  # Default to 120 retries if not provided
    local interval=${3:-1}   # Default to 1 second interval if not provided
    local counter=0

    until [ $counter -ge $retries ]
    do
        # Use curl to check if the service is available
        if curl -f -s -o /dev/null "$url"; then
            echo "Service is available"
            return 0
        fi
        echo "Service is not available yet. Retrying in $interval seconds..."
        sleep $interval
        counter=$((counter+1))
    done

    echo "Service did not become available within the specified retries"
    exit 1
}

# Call the function with the URL of the service
wait_for_service "$1"

# Execute the Python command
exec python cli/build.py --data_path "/data/raw/ex_metadata.csv" --model_name "text-embedding-ada-002" --openai_api_key "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"
