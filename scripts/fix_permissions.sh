#!/bin/bash
# Fix directory permissions for ΨQRH project
# Run with: sudo ./scripts/fix_permissions.sh

set -e

echo "🔧 Fixing ΨQRH Directory Permissions"
echo "====================================="
echo ""

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_GROUP=$(id -gn $ACTUAL_USER)

echo "Setting ownership to: $ACTUAL_USER:$ACTUAL_GROUP"
echo ""

# Fix models directory
if [ -d "models" ]; then
    echo "📁 Fixing models/ directory..."
    chown -R $ACTUAL_USER:$ACTUAL_GROUP models/
    chmod -R u+rw models/
    echo "✅ models/ fixed"
fi

# Fix any other root-owned directories
for dir in tmp logs data .cache; do
    if [ -d "$dir" ]; then
        echo "📁 Fixing $dir/ directory..."
        chown -R $ACTUAL_USER:$ACTUAL_GROUP "$dir/" 2>/dev/null || true
        chmod -R u+rw "$dir/" 2>/dev/null || true
        echo "✅ $dir/ fixed"
    fi
done

# Create missing directories with correct permissions
echo ""
echo "📁 Creating missing directories..."
for dir in models/pretrained models/finetuned models/checkpoints tmp logs; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        chown $ACTUAL_USER:$ACTUAL_GROUP "$dir"
        chmod 755 "$dir"
        echo "✅ Created $dir/"
    fi
done

echo ""
echo "✅ Permissions fixed successfully!"
echo ""
echo "Directories now owned by: $ACTUAL_USER:$ACTUAL_GROUP"
ls -la models/ | head -5