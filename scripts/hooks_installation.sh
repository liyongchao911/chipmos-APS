#!/bin/sh
if ! test -d .git; then
    echo "Execute hook installation on the top-level directory which is with .git directory"
    exit 1
fi

ln -sf ../../scripts/pre-commit.sh .git/hooks/pre-commit || exit 1
chmod +x .git/hooks/pre-commit || exit 2

ln -sf ../../scripts/commit-msg.sh .git/hooks/commit-msg || exit 1
chmod +x .git/hooks/commit-msg || 2

touch .git/hooks/installed

echo "git hook is installed successful"
