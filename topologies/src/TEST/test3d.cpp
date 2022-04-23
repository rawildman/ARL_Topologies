/*
 * ARL_Topologies - An extensible topology optimization program
 *
 * Written in 2017 by Raymond A. Wildman <raymond.a.wildman.civ@mail.mil>
 * This project constitutes a work of the United States Government and is not
 * subject to domestic copyright protection under 17 USC Sec. 105.
 * Release authorized by the US Army Research Laboratory
 *
 * To the extent possible under law, the author(s) have dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along
 * with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 *
 */

// Questions?
// Contact: Raymond Wildman, raymond.a.wildman.civ@mail.mil

#define CATCH_CONFIG_MAIN

#include "topoptuniverse.h"
#include "topoptrep.h"
#include "catch2/catch_all.hpp"
#include <vector>
#include <fstream>

using namespace Topologies;

void printRes(const TopOptRep& res, const std::string& fileName)
{
	std::vector<double> realRep;
	res.getRealRep(realRep);
	std::ofstream outFile(fileName.c_str());
	outFile << realRep.size();
	outFile.precision(16);
	for(auto it = realRep.begin(); it != realRep.end(); ++it)
		outFile << " " << *it;
	std::cout << std::endl;
}

std::vector<double> readRes(const std::string& fileName)
{
	std::ifstream inFile(fileName.c_str());
	std::size_t n;
	inFile >> n;
	std::vector<double> outVec(n);
	for(std::size_t k = 0; k < n; ++k)
		inFile >> outVec[k];
	return outVec;
}

TEST_CASE("Testing 3D voxel, element-based densities","[Topologies]")
{
	char progName[] = "test3d";
	char fileName[] = "testvox.xml";
	char* argv[] = {progName, fileName};
	TopOptUniverse toProblem(2, argv);
	toProblem.runProblem();
	std::unique_ptr<TopOptRep> result(toProblem.getResult());
	// Load result from file (generated previously)
	std::vector<double> resVec = readRes("voxres.txt");
	// Get optimization values
	std::vector<double> realRep;
	result->getRealRep(realRep);
	// Compare
	REQUIRE(realRep.size() == resVec.size());
	for(std::size_t k = 0; k < realRep.size(); ++k)
		REQUIRE(realRep[k] == Catch::Approx(resVec[k]));
}

TEST_CASE("Testing 3D voxel, element-based densities and tetrahedral elements","[Topologies]")
{
	char progName[] = "test3d";
	char fileName[] = "testvoxtet.xml";
	char* argv[] = {progName, fileName};
	TopOptUniverse toProblem(2, argv);
	toProblem.runProblem();
	std::unique_ptr<TopOptRep> result(toProblem.getResult());
	// Load result from file (generated previously)
	std::vector<double> resVec = readRes("voxrestet.txt");
	// Get optimization values
	std::vector<double> realRep;
	result->getRealRep(realRep);
	// Compare
	REQUIRE(realRep.size() == resVec.size());
	for(std::size_t k = 0; k < realRep.size(); ++k)
		REQUIRE(realRep[k] == Catch::Approx(resVec[k]));
}

TEST_CASE("Testing 3D heaviside with voxel, node-based densities","[Topologies]")
{
	char progName[] = "test2d";
	char fileName[] = "testheavi3.xml";
	char* argv[] = {progName, fileName};
	TopOptUniverse toProblem(2, argv);
	toProblem.runProblem();
	std::unique_ptr<TopOptRep> result(toProblem.getResult());
//	printRes(*result,"heavi3res.txt"); // Generates result file
	// Load result from file (generated previously)
	std::vector<double> resVec = readRes("heavi3res.txt");
	// Get optimization values
	std::vector<double> realRep;
	result->getRealRep(realRep);
	// Compare
	REQUIRE(realRep.size() == resVec.size());
	for(std::size_t k = 0; k < realRep.size(); ++k)
		REQUIRE(realRep[k] == Catch::Approx(resVec[k]));
}

TEST_CASE("Testing 3D tet mesh with a fixed block","[Topologies]")
{
	char progName[] = "test3d";
	char fileName[] = "testmesh3_2blocks.xml";
	char* argv[] = {progName, fileName};
	TopOptUniverse toProblem(2, argv);
	toProblem.runProblem();
	std::unique_ptr<TopOptRep> result(toProblem.getResult());
	// Load result from file (generated previously)
	std::vector<double> resVec = readRes("mesh3_2blocksres.txt");
	// Get optimization values
	std::vector<double> realRep;
	result->getRealRep(realRep);
	// Compare
	REQUIRE(realRep.size() == resVec.size());
	for(std::size_t k = 0; k < realRep.size(); ++k)
		REQUIRE(realRep[k] == Catch::Approx(resVec[k]));
}

TEST_CASE("Testing 3D tet mesh with a fixed block and Heaviside proj.","[Topologies]")
{
	char progName[] = "test3d";
	char fileName[] = "testheavimesh3_2blocks.xml";
	char* argv[] = {progName, fileName};
	TopOptUniverse toProblem(2, argv);
	toProblem.runProblem();
	std::unique_ptr<TopOptRep> result(toProblem.getResult());
	// Load result from file (generated previously)
	std::vector<double> resVec = readRes("heavimesh3_2blocksres.txt");
	// Get optimization values
	std::vector<double> realRep;
	result->getRealRep(realRep);
	// Compare
	REQUIRE(realRep.size() == resVec.size());
	for(std::size_t k = 0; k < realRep.size(); ++k)
		REQUIRE(realRep[k] == Catch::Approx(resVec[k]));
}

