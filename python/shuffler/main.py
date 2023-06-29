# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expresus or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

from fcp.demo import server

POPULATION_NAME = 'test/population'

if __name__ == '__main__':
  server = server.InProcessServer(
      population_name=POPULATION_NAME,
      host='0.0.0.0',
      port=55555,
      address_family=None)
  server_thread = threading.Thread(
      target=server.serve_forever, daemon=True)
  server_thread.start()
  print("Federated Compute server running on %s:%s", server.server_name, server.server_port)

  server.serve_forever()