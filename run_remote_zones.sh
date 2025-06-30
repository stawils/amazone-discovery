#!/bin/bash

# Script to run archaeological discovery on remote zones
# Each zone runs separately with checkpoint 2 and no OpenAI
# This script dynamically extracts all zone IDs from the config.py file

echo "🚀 Starting Amazon Archaeological Discovery for Remote Zones"
echo "============================================================"

# Extract all zone IDs from the TARGET_ZONES dictionary only
echo "📋 Extracting zone IDs from config.py..."
zones=($(awk '/TARGET_ZONES = {/,/^}/ { if ($0 ~ /"[a-zA-Z0-9_]*": TargetZone/) { match($0, /"([a-zA-Z0-9_]*)"/, arr); print arr[1] } }' "$(dirname "$0")/src/core/config.py"))

# Parse script arguments for special flags
DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" || "$arg" == "-d" ]]; then
        DRY_RUN=true
    fi
done

# Function to run a single zone
run_zone() {
    local zone_id=$1
    echo ""
    echo "🎯 Processing Zone: $zone_id"
    echo "⏰ Started at: $(date)"
    echo "----------------------------------------"
    
    # Run the pipeline with checkpoint 2
    python main.py --checkpoint 2 --zone "$zone_id" --no-openai
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Zone $zone_id completed successfully"
    else
        echo "❌ Zone $zone_id failed with exit code $exit_code"
    fi
    
    echo "⏰ Finished at: $(date)"
    echo "----------------------------------------"
}

# Main execution
echo "📋 Zones to process: ${#zones[@]}"

# Check if any zones were found
if [ ${#zones[@]} -eq 0 ]; then
    echo "❌ Error: No zones found in config.py. Please check the config file."
    exit 1
fi

# Display all zones to be processed
for zone in "${zones[@]}"; do
    echo "   - $zone"
done
echo ""

# Exit early if dry-run flag is set
if $DRY_RUN; then
    echo "📝 Dry run selected. Exiting without running analysis."
    exit 0
fi

start_time=$(date)
echo "🕐 Total run started at: $start_time"
echo ""

# Process each zone
for zone_id in "${zones[@]}"; do
    run_zone "$zone_id"
    
    # Add a small delay between runs
    echo "⏳ Waiting 5 seconds before next zone..."
    sleep 5
done

end_time=$(date)
echo ""
echo "🏁 All zones completed!"
echo "🕐 Started:  $start_time"
echo "🕑 Finished: $end_time"
echo ""
echo "📊 Results can be found in the results/ directory"
echo "   Each zone will have its own run_YYYYMMDD_HHMMSS_<zone_id> directory"