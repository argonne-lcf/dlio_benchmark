"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class Confidence(Enum):
    HIGH = "high"      # >100 events, low variance — emit as-is
    MEDIUM = "medium"  # 10-100 events or moderate variance — note in YAML comment
    LOW = "low"        # <10 events or high variance — warn user to verify


@dataclass
class ParameterEstimate(Generic[T]):
    value: T
    confidence: Confidence
    source: str = ""   # human-readable explanation of how value was inferred

    def comment(self) -> str | None:
        if self.confidence == Confidence.HIGH:
            return None
        tag = f"[{self.confidence.value.upper()} confidence"
        if self.confidence == Confidence.LOW:
            tag += " — verify"
        tag += "]"
        return tag
