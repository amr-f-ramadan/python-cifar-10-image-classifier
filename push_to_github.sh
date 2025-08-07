#!/bin/bash

# CIFAR-10 Repository Setup Script
# Run this script after creating the remote repository on GitHub

echo "🚀 CIFAR-10 Image Classifier - GitHub Repository Setup"
echo "===================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run this from the project root."
    exit 1
fi

# Get GitHub username
read -p "📝 Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ GitHub username is required"
    exit 1
fi

# Repository details
REPO_NAME="python-cifar-10-image-classifier"
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo ""
echo "🔗 Repository URL: $REMOTE_URL"
echo ""

# Add remote origin
echo "🔗 Adding remote origin..."
git remote add origin $REMOTE_URL

# Verify remote was added
echo "✅ Remote added successfully:"
git remote -v

echo ""
echo "📤 Pushing to GitHub..."

# Push to main branch
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Successfully pushed to GitHub!"
    echo ""
    echo "🌐 Your repository is now available at:"
    echo "   https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    echo ""
    echo "🎯 Repository features:"
    echo "   ✅ CIFAR-10 Image Classifier with dual CNN architectures"
    echo "   ✅ Automated conda environment setup"
    echo "   ✅ Cross-platform GPU acceleration support"
    echo "   ✅ 78%+ accuracy achieved"
    echo "   ✅ Production-ready code and documentation"
    echo ""
    echo "🚀 Others can now clone and run with:"
    echo "   git clone $REMOTE_URL"
    echo "   cd $REPO_NAME"
    echo "   chmod +x setup_environment.sh && ./setup_environment.sh"
else
    echo ""
    echo "❌ Push failed. Please check:"
    echo "   1. Repository exists on GitHub"
    echo "   2. You have write permissions"
    echo "   3. Your GitHub credentials are correct"
    echo ""
    echo "💡 You may need to authenticate with GitHub:"
    echo "   - Use GitHub CLI: gh auth login"
    echo "   - Or use personal access token as password"
fi
