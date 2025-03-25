from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import logging
from ..agents import WorkflowAgent, AutonomousAgent