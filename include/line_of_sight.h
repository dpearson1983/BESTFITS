#ifndef _LINE_OF_SIGHT_H_
#define _LINE_OF_SIGHT_H_

#include <vector>
#include "tpods.h"

void get_A0(std::vector<double> &dr, std::vector<double> &A_0, vec3<int> N);

void get_A2(std::vector<double> &dr, std::vector<double> &A_2, vec3<int> N, vec3<double> L, vec3<double> r_min);

void get_A4(std::vector<double> &dr, std::vector<double> &A_4, vec3<int> N, vec3<double> L, vec3<double> r_min);

#endif
