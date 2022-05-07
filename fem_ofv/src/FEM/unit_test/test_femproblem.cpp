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

#define CATCH_CONFIG_MAIN

#include "linearelasticproblem.h"
#include "laplaceproblem.h"
#include "REP/tomesh.h"
#include "catch2/catch_all.hpp"

#include <iostream>

using namespace Topologies;

namespace{
	const std::function<double(double,double)> identity = [](double val, double){return val;};
	const std::function<double(double,double)> permittivity = [](double mat, double opt) {
		return (mat - 1.0) * opt + 1.0;};
	const std::vector<MaterialFunction> linearElasticOptimizationParameters = {identity, 
		std::multiplies<double>(), std::multiplies<double>()};
	const std::vector<MaterialFunction> laplaceOptimizationParameters = {identity, 
		std::multiplies<double>(), std::multiplies<double>(), permittivity};
}

TEST_CASE("Testing LinearElasticProblem in 2D", "[LinearElasticProblem]")
{
	Point_2_base p0(0., 0.), p1(1., 0.), p2(1., 1.), p3(0., 1.);
	std::vector<Point_2_base> nodeVec = {p0, p1, p2, p3};
	std::vector<double> mats(3, 1.);
	GenericMaterial gm(mats);
	// Set up boundary conditions
	ExoBC supp1(2);
	supp1.isSupport = true;
	supp1.xsup = supp1.ysup = true;
	supp1.nodeIDVec = {0};
	ExoBC supp2(2);
	supp2.isSupport = true;
	supp2.xsup = true;
	supp2.ysup = false;
	supp2.nodeIDVec = {3};
	ExoBC load(2);
	load.isSupport = false;
	load.loadVec = Point_3_base(0.5, 0., 0.);
	load.nodeIDVec = {1, 2};
	std::vector<ExoBC> bcVec = {supp1, supp2, load};
	// Solution
	Eigen::VectorXd chkVec(5);
	chkVec << 0.4, 0.4, 0., -0.1, -0.1;
	SECTION("Quad elements")
	{
		// Set up mesh
		std::vector<std::vector<std::size_t>> connVec(1);
		std::vector<std::size_t> pts1 = {0, 1, 2, 3};
		connVec[0] = pts1;
		const std::vector<double> optVals = {1.0};
		const TOMesh2D mesh(nodeVec, connVec, optVals);
		// Set up & solve FE problem
		LinearElasticProblem testFE(mesh, gm, linearElasticOptimizationParameters);
		testFE.changeBoundaryConditionsTo(bcVec);
		REQUIRE(testFE.validRun());
		REQUIRE(chkVec.isApprox(testFE.getDisplacement(), 1e-14));
		const std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(0.4));
		REQUIRE(c.second);
		
		// Backwards difference approx to check gradient
		const std::vector<double> gc = testFE.gradCompliance(mesh);
		const std::vector<double> optValsBD = {1.0 - 1e-8};
		const TOMesh2D meshBD(nodeVec, connVec, optValsBD);
		LinearElasticProblem testFEBD(meshBD, gm, linearElasticOptimizationParameters);
		testFEBD.changeBoundaryConditionsTo(bcVec);
		const std::pair<double, bool> cbd = testFEBD.computeCompliance();
		const double gcBd = (c.first - cbd.first)/1e-8;
		REQUIRE(gc.size() == 1);
		REQUIRE(gc[0] == Catch::Approx(gcBd));
	}
	SECTION("Tri elements")
	{
		// Set up mesh
		std::vector<std::vector<std::size_t>> connVec(2);
		connVec[0] = {0, 1, 2};
		connVec[1] = {0, 2, 3};
		std::vector<double> optVals(2, 1.);
		TOMesh2D mesh(nodeVec, connVec, optVals);
		// Set up & solve FE problem
		LinearElasticProblem testFE(mesh, gm, linearElasticOptimizationParameters);
		testFE.changeBoundaryConditionsTo(bcVec);
		REQUIRE(testFE.validRun());
		REQUIRE(chkVec.isApprox(testFE.getDisplacement(), 1e-14));
		std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(0.4));
		REQUIRE(c.second);
	}
}

TEST_CASE("Testing LinearElasticProblem in 3D", "[LinearElasticProblem]")
{
	Point_3_base p0(0., 0., 0.), p1(1., 0., 0.), p2(0., 1., 0), p3(1., 1., 0.);
	Point_3_base p4(0., 0., 1.), p5(1., 0., 1.), p6(0., 1., 1), p7(1., 1., 1.);
	std::vector<Point_3_base> nodeVec = {p0, p1, p2, p3, p4, p5, p6, p7};
	ExoBC supp1(3);
	supp1.isSupport = true;
	supp1.xsup = supp1.ysup = supp1.zsup = true;
	supp1.nodeIDVec = {0};
	// SUPP2
	ExoBC supp2(3);
	supp2.isSupport = true;
	supp2.xsup = false;
	supp2.ysup = supp2.zsup = true;
	supp2.nodeIDVec = {1};
	// Supp3
	ExoBC supp3(3);
	supp3.isSupport = true;
	supp3.ysup = false;
	supp3.xsup = supp3.zsup = true;
	supp3.nodeIDVec = {2};
	// Supp4
	ExoBC supp4(3);
	supp4.isSupport = true;
	supp4.xsup = supp4.ysup = false;
	supp4.zsup = true;
	supp4.nodeIDVec = {3};
	// Loads
	ExoBC load1(3);
	load1.isSupport = false;
	load1.loadVec = Point_3_base(0., 0., 0.1666666666666666666667);
	load1.nodeIDVec = {4, 7};
	ExoBC load2(3);
	load2.isSupport = false;
	load2.loadVec = Point_3_base(0., 0., 0.33333333333333333333333);
	load2.nodeIDVec = {5, 6};
	ExoBC load3(3);
	load3.isSupport = false;
	load3.loadVec = Point_3_base(0., 0., 0.25);
	load3.nodeIDVec = {4, 5, 6, 7};
	std::vector<ExoBC> bcVec = {supp1, supp2, supp3, supp4};
	std::vector<double> mats(3, 1.);
	GenericMaterial gm(mats);
	// Solution
	Eigen::VectorXd chkVec(16);
	chkVec << -0.1, -0.1, 0., -0.1, 0., -0.1, -0.1, -0.1, 0., 0., -0.1, -0.1, 0.4, 0.4, 0.4, 0.4;
	SECTION("Tet elements")
	{
		std::vector<std::vector<std::size_t>> connVec(6);
		connVec[0] = {4, 0, 2, 1};
		connVec[1] = {6, 4, 2, 1};
		connVec[2] = {6, 5, 4, 1};
		connVec[3] = {6, 3, 5, 1};
		connVec[4] = {6, 2, 3, 1};
		connVec[5] = {6, 7, 5, 3};
		std::vector<double> optVals(6, 1.);
		TOMesh3D mesh(nodeVec, connVec, optVals);
		LinearElasticProblem testFE(mesh, gm, linearElasticOptimizationParameters);
		bcVec.push_back(load1);
		bcVec.push_back(load2);
		testFE.changeBoundaryConditionsTo(bcVec);
		REQUIRE(testFE.validRun());
		REQUIRE(chkVec.isApprox(testFE.getDisplacement(), 1e-6));
		std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(0.4));
		REQUIRE(c.second);
	}

	SECTION("Hex elements")
	{
		std::vector<std::vector<std::size_t>> connVec(1);
		connVec[0] = {0, 1, 3, 2, 4, 5, 7, 6};
		std::vector<double> optVals(1, 1.);
		TOMesh3D mesh(nodeVec, connVec, optVals);
		LinearElasticProblem testFE(mesh, gm, linearElasticOptimizationParameters);
		bcVec.push_back(load3);
		testFE.changeBoundaryConditionsTo(bcVec);
		const Eigen::VectorXd &sol = testFE.getDisplacement();
		REQUIRE(testFE.validRun());
		REQUIRE(chkVec.isApprox(testFE.getDisplacement(), 1e-6));
		std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(0.4));
		REQUIRE(c.second);
	}
}

TEST_CASE("Testing LaplaceProblem in 2D", "[LaplaceProblem]")
{
	const Point_2_base p00(0., 0.0), p10(10., 0.0), p20(20., 0.0);
	const Point_2_base p01(0., 10.), p11(10., 10.), p21(20., 10.0);
	const Point_2_base p02(0., 20.), p12(10., 20.), p22(20., 20.0);
	const std::vector<Point_2_base> nodeVec = {p00, p10, p20, p01, p11, p21, p02, p12, p22};
	const std::vector<double> mats(4, 21.0);
	const GenericMaterial gm(mats);
	// Set up boundary conditions
	ExoBC supp(2);
	supp.isSupport = true;
	supp.xsup = supp.ysup = supp.zsup = true;
	supp.nodeIDVec = {0, 1, 2};
	ExoBC load(2);
	load.isSupport = false;
	load.loadVec = Point_3_base(1.0, 0., 0.);
	load.nodeIDVec = {6, 7, 8};
	const std::vector<ExoBC> bcVec = {supp, load};
	// Solution
	Eigen::VectorXd chkVec(3);
	chkVec << 0.5, 0.5, 0.5;
	SECTION("Quad elements")
	{
		// Set up mesh
		const std::vector<std::vector<std::size_t>> connVec =
			{{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
		const std::vector<double> optVals(4, 1.0);
		//const std::vector<double> optVals = {0.25, 0.25, 0.75, 0.75};
		const TOMesh2D mesh(nodeVec, connVec, optVals);
		// Set up & solve FE problem
		LaplaceProblem testFE(mesh, gm, laplaceOptimizationParameters);
		testFE.changeBoundaryConditionsTo(bcVec);
		REQUIRE(testFE.validRun());
		REQUIRE(chkVec.isApprox(testFE.getDisplacement(), 1e-14));
		//std::cout << "b = \n" << testFE.getForce() << std::endl;

		const std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(21.0));
		REQUIRE(c.second);

		// Backwards difference approx to check gradient
		const std::vector<double> gc = testFE.gradCompliance(mesh);
		for(int k = 0; k < 4; ++k)
		{
			constexpr double h = 1e-8;
			std::vector<double> optValsBD = optVals;
			optValsBD[k] = optVals[k] - h;
			const TOMesh2D meshBD(nodeVec, connVec, optValsBD);
			LaplaceProblem testFEBD(meshBD, gm, laplaceOptimizationParameters);
			testFEBD.changeBoundaryConditionsTo(bcVec);
			const std::pair<double, bool> cbd = testFEBD.computeCompliance();
			const double gcBd = (c.first - cbd.first)/h;
			REQUIRE(gc.size() == 4);
			std::cout << "k = " << k << ", gcBd = " << gcBd << ", gc = " << gc[k] << std::endl;
			REQUIRE(gc[k] == Catch::Approx(gcBd));
		}
	}
	SECTION("Tri elements")
	{
		// Set up mesh
		const std::vector<std::vector<std::size_t>> connVec =
			{{0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}, 
			 {3, 4, 7}, {3, 7, 6}, {4, 5, 8}, {4, 8, 7}};
		const std::vector<double> optVals(8, 0.5);
		const TOMesh2D mesh(nodeVec, connVec, optVals);
		// Set up & solve FE problem
		LaplaceProblem testFE(mesh, gm, laplaceOptimizationParameters);
		testFE.changeBoundaryConditionsTo(bcVec);
		REQUIRE(testFE.validRun());
		REQUIRE(chkVec.isApprox(testFE.getDisplacement(), 1e-14));

		const std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(11.0));
		REQUIRE(c.second);

		// Backwards difference approx to check gradient
		const std::vector<double> gc = testFE.gradCompliance(mesh);
		for(int k = 0; k < 8; ++k)
		{
			constexpr double h = 1e-8;
			std::vector<double> optValsBD = optVals;
			optValsBD[k] = optVals[k] - h;
			const TOMesh2D meshBD(nodeVec, connVec, optValsBD);
			LaplaceProblem testFEBD(meshBD, gm, laplaceOptimizationParameters);
			testFEBD.changeBoundaryConditionsTo(bcVec);
			const std::pair<double, bool> cbd = testFEBD.computeCompliance();
			const double gcBd = (c.first - cbd.first)/h;
			REQUIRE(gc.size() == 8);
			std::cout << "k = " << k << ", gcBd = " << gcBd << ", gc = " << gc[k]  << std::endl;
			REQUIRE(gc[k] == Catch::Approx(gcBd));
		}
	}
}

TEST_CASE("Testing LaplaceProblem in 2D larger", "[LaplaceProblem]")
{
	const Point_2_base p00(0., 0.0), p10(10., 0.0), p20(20., 0.0);
	const Point_2_base p01(0., 10.), p11(10., 10.), p21(20., 10.0);
	const Point_2_base p02(0., 20.), p12(10., 20.), p22(20., 20.0);
	const Point_2_base p03(0., 30.), p13(10., 30.), p23(20., 30.0);
	const Point_2_base p04(0., 40.), p14(10., 40.), p24(20., 40.0);
	const std::vector<Point_2_base> nodeVec = {p00, p10, p20, 
		p01, p11, p21, 
		p02, p12, p22,
		p03, p13, p23,
		p04, p14, p24};
	const std::vector<double> mats(4, 41.0);
	const GenericMaterial gm(mats);
	// Set up boundary conditions
	ExoBC supp(2);
	supp.isSupport = true;
	supp.xsup = supp.ysup = supp.zsup = true;
	supp.nodeIDVec = {0, 1, 2};
	ExoBC load(2);
	load.isSupport = false;
	load.loadVec = Point_3_base(1.0, 0., 0.);
	load.nodeIDVec = {12, 13, 14};
	const std::vector<ExoBC> bcVec = {supp, load};
	SECTION("Quad elements")
	{
		// Set up mesh
		const std::vector<std::vector<std::size_t>> connVec =
			{{0, 1, 4, 3}, {1, 2, 5, 4}, 
			 {3, 4, 7, 6}, {4, 5, 8, 7},
			 {6, 7, 10, 9}, {7, 8, 11, 10},
			 {9, 10, 13, 12}, {10, 11, 14, 13}};
		/*const std::vector<double> optVals = {0.125, 0.125, 
																				 0.375, 0.375,
																				 0.625, 0.625,
																				 0.875, 0.875};*/
		const std::vector<double> optVals(8, 1.0);
		const TOMesh2D mesh(nodeVec, connVec, optVals);
		// Set up & solve FE problem
		LaplaceProblem testFE(mesh, gm, laplaceOptimizationParameters);
		testFE.changeBoundaryConditionsTo(bcVec);
		REQUIRE(testFE.validRun());
		const std::pair<double, bool> c = testFE.computeCompliance();
		REQUIRE(c.first == Catch::Approx(20.5));
		REQUIRE(c.second);
	}
}
