/* SPDX-License-Identifier: Apache-2.0 */
// Retired entrypoint retained so old build recipes report the deprecation clearly.
#include <iostream>

int main()
{
  std::cerr << "FAVOR is retired. Use CAGRA-Base, CAGRA-Retain, or CAGRA-NaviX.\n";
  return 1;
}
