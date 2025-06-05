#!/bin/bash
# Set the correct library path to use our compiled sentencepiece
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Run the server with the specified model
./target/release/llmserver-rs "$@"
