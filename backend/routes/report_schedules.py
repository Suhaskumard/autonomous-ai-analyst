"""Scheduled reports — Phase 10. See report_schedule.py for what runs them."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth import Principal, current_principal, require_admin, require_owned_run
from report_schedule import create_schedule, delete_schedule, list_schedules, run_due

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateScheduleRequest(BaseModel):
    recipients: list[str] = Field(default_factory=list)
    interval: str = "weekly"


@router.post("/runs/{run_key}/report-schedules")
def create_report_schedule(
    run_key: str, request: CreateScheduleRequest, principal: Principal = Depends(current_principal)
):
    require_owned_run(run_key, principal)
    return create_schedule(run_key, principal, request.recipients, request.interval)


@router.get("/runs/{run_key}/report-schedules")
def list_report_schedules(run_key: str, principal: Principal = Depends(current_principal)):
    require_owned_run(run_key, principal)
    return {"schedules": list_schedules(run_key, principal)}


@router.delete("/report-schedules/{schedule_id}")
def delete_report_schedule(schedule_id: str, principal: Principal = Depends(current_principal)):
    delete_schedule(schedule_id, principal)
    return {"schedule_id": schedule_id, "deleted": True}


@router.post("/report-schedules/run-due")
def report_schedules_run_due(principal: Principal = Depends(require_admin)):
    """What an operator's cron entry calls. Same posture as /admin/retention/purge."""
    return run_due()
