#!/usr/bin/env bash
# Provision missing test-* hosts in us-west-2 cloned from test-c8a's config.
# Idempotent: skips hosts that already exist (anywhere).
set -euo pipefail

cd "$(dirname "$0")"

REGION=us-west-2
SUBNET=subnet-a5141ec0
SG=sg-025b1178
KEY=AWS-key-Oregon
AMI_X86=ami-029a761f237195c2c
AMI_ARM=ami-0a2a049c945b84826

# user-data: install gcc/cmake/git/make + a marker file for the runner to poll.
read -r -d '' USER_DATA <<'EOF' || true
#!/bin/bash
set -e
exec >/var/log/bootstrap.log 2>&1
dnf install -y gcc gcc-c++ cmake git make rsync tar gzip || \
  yum install -y gcc gcc-c++ cmake git make rsync tar gzip
touch /home/ec2-user/.bootstrap_done
chown ec2-user:ec2-user /home/ec2-user/.bootstrap_done
EOF
USER_DATA_B64=$(printf '%s' "$USER_DATA" | base64)

launch() {
    local alias=$1 itype=$2 arch=$3
    local ami
    case "$arch" in
        x86_64) ami=$AMI_X86 ;;
        arm64)  ami=$AMI_ARM ;;
        *) echo "unknown arch $arch" >&2; return 1 ;;
    esac
    echo "[$alias] launching $itype ($arch, $ami)..." >&2
    aws ec2 run-instances --region "$REGION" \
        --image-id "$ami" \
        --instance-type "$itype" \
        --key-name "$KEY" \
        --subnet-id "$SUBNET" \
        --security-group-ids "$SG" \
        --user-data "$USER_DATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$alias},{Key=sweep,Value=uarch}]" \
        --query 'Instances[0].InstanceId' --output text
}

# Read hosts.tsv, skip comments/blank, skip existing=yes.
while IFS=$'\t ' read -r alias itype arch family year uarch region existing; do
    [[ -z "${alias:-}" || "${alias:0:1}" == "#" ]] && continue
    [[ "$existing" == "yes" ]] && { echo "[$alias] existing, skipping" >&2; continue; }
    launch "$alias" "$itype" "$arch"
done < <(awk 'NF && $1 !~ /^#/' hosts.tsv)

echo "done; poll with: aws ec2 describe-instances --region $REGION --filters Name=tag:sweep,Values=uarch --query 'Reservations[].Instances[].[Tags[?Key==\`Name\`].Value|[0],State.Name,PublicDnsName]' --output table"
