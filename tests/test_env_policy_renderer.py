# -*- coding: utf-8 -*-
"""Unit tests for env policy runtime rendering."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core.env_policy_renderer import render_env_exports  # noqa: E402


def test_value_template_expands_dollar_placeholders():
    commands = render_env_exports(
        [
            {
                "name": "HCCL_IF_IP",
                "mode": "force_override",
                "value_template": "${current_ip}",
                "applies_when": {"deployment_mode": "dp_deployment"},
            },
            {
                "name": "GLOO_SOCKET_IFNAME",
                "mode": "force_override",
                "value_template": "${network_interface}",
                "applies_when": {"deployment_mode": "dp_deployment"},
            },
        ],
        {
            "deployment_mode": "dp_deployment",
            "current_ip": "10.1.2.3",
            "network_interface": "eth1",
        },
    )

    assert commands == [
        "export HCCL_IF_IP=10.1.2.3",
        "export GLOO_SOCKET_IFNAME=eth1",
    ]


def test_applies_when_filters_non_matching_policy():
    commands = render_env_exports(
        [
            {
                "name": "HCCL_IF_IP",
                "mode": "force_override",
                "value_template": "${current_ip}",
                "applies_when": {"deployment_mode": "ray"},
            }
        ],
        {"deployment_mode": "dp_deployment", "current_ip": "10.1.2.3"},
    )

    assert commands == []


def test_literal_shell_expressions_are_preserved():
    commands = render_env_exports(
        [
            {
                "name": "VLLM_HOST_IP",
                "mode": "force_override",
                "value_template": "${POD_IP:-${RANK_IP:-$(python3 -c 'print(1)')}}",
                "applies_when": {"deployment_mode": "ray"},
            },
            {
                "name": "HCCL_IF_IP",
                "mode": "force_override",
                "value_template": "$VLLM_HOST_IP",
                "applies_when": {"deployment_mode": "ray"},
            },
        ],
        {"deployment_mode": "ray"},
    )

    assert commands == [
        "export VLLM_HOST_IP=${POD_IP:-${RANK_IP:-$(python3 -c 'print(1)')}}",
        "export HCCL_IF_IP=$VLLM_HOST_IP",
    ]
