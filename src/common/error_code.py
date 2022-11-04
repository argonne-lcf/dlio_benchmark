"""
   Copyright 2021 UChicago Argonne, LLC

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


class ErrorCode(object):
    def __init__(self, error_code, error_message):
        self.error_code_ = error_code
        self.error_message_ = error_message

    def __repr__(self):
        return {'error_code': self.error_code_, 'error_message': self.error_message_}

    def __str__(self):
        return self.error_message_.format(self.error_code_)


class ErrorCodes:
    EC0000 = {0, "SUCCESSFUL"}
    EC1000 = {1000, "ERROR: Incorrect Computation Type"}
    EC1001 = {1001, "ERROR: Incorrect Format Type"}
    EC1002 = {1002, "ERROR: Invalid Parameter Combination"}
