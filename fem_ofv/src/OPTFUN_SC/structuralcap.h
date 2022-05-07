/*
 * ARL_Topologies - An extensible topology optimization program
 * 
 * Written in 2022l by Raymond A. Wildman
 * 
 * To the extent possible under law, the author(s) have dedicated all copyright 
 * and related and neighboring rights to this software to the public domain 
 * worldwide. This software is distributed without any warranty.
 * 
 * You should have received a copy of the CC0 Public Domain Dedication along 
 * with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>. 
 * 
 */

#ifndef STRUCTURALCAP_H
#define STRUCTURALCAP_H

#include <string>
#include <memory>
#include "OBJFUN/topoptobjfun.h"
#include "REP/cgal_types.h"
#include "IO/inputloader.h"
#include "boundarycondition.h"
#include "loadcondition.h"
#include "femproblem.h"

namespace Topologies{
	class TopOptRep;
	class TOMesh;
}
class LinearElasticProblem;
class LaplaceProblem;

struct FEMProblems
{
	std::unique_ptr<LinearElasticProblem> upLinearElasticProblem;
	std::unique_ptr<LaplaceProblem> upLaplaceProblem;
};

//! Interface between finite element solver and TopOptObjFun base class
/*!
 * This class is derived from TopOptObjFun and interfaces between it and FEMProblem.  
 * FEMProblem is a static, linear elastic solver with various element types.  
 * This class will construct an FEMProblem object, solve a given problem, and return 
 * the compliance which is the dot product of the displacement and load vectors.
 */
class StructuralCap : public Topologies::TopOptObjFun
{
public:
	//! Constructor which takes the name of an input file.
	StructuralCap(const std::string& inpFileName);

	void printResult(const Topologies::TopOptRep& inTOR, const std::string& fileName) const override {}
	std::pair<double, bool> operator()(const Topologies::TopOptRep& inTOR) const override;
	void f(const Topologies::TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const override;
	void c(const Topologies::TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const override;
  void g(const Topologies::TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const override;
	void gc(const Topologies::TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const override;
	void fAndG(const Topologies::TopOptRep& inTOR, std::pair<std::vector<double>, bool>& fRes,
						 std::pair<std::vector<double>, bool>& gRes) const override; 
private:
  std::vector<ExoBC> generateExoBCVec(const Topologies::TOMesh* const inMesh, const std::vector<BoundaryCondition>& bcVec, const std::vector<LoadCondition<double>>& lcVec) const;
	std::vector<ExoBC> generateExoBCVecLinearElastic(const Topologies::TOMesh* const inMesh, std::size_t kLoad) const;
	std::vector<ExoBC> generateExoBCVecLaplace(const Topologies::TOMesh* const inMesh, std::size_t kLoad) const;
	FEMProblems setupAndSolveFEM(const Topologies::TopOptRep& inTOR, std::size_t kLoad) const;
private:
	unsigned dim;
	Topologies::GenericMaterial baseMat;
	double maxDisplacement, volfrac, weight, complianceScale, capacitanceScale;
	std::vector<BoundaryCondition> bcVecLinearElastic;
	std::vector<BoundaryCondition> bcVecLaplace;
	std::vector<std::vector<LoadCondition<double>>> lcVVLinearElastic;
	std::vector<std::vector<LoadCondition<double>>> lcVVLaplace;
};

// Factory load/destroy functions to interface with optimization code
extern "C" Topologies::TopOptObjFun* create(const std::string& inpFileName) 
{
	return new StructuralCap(inpFileName);
}

extern "C" void destroy(Topologies::TopOptObjFun* p) 
{
	delete p;
}
// Functions defined in tofor:
// typedef TopOptObjFun* create_toof(const std::string& inpFileName);
// typedef void destroy_toof(TopOptObjFun*);

//! XML input parser for the FEM problem
class FEMInputLoader : public Topologies::InputLoader::InputParser
{
public:
	void parse(const pugi::xml_document& xmldoc) override;
	void parse(const pugi::xml_node& rootNode) override;
	
	//! Returns Young's modulus
	double getYoungsMod() const {return E;}
	//! Returns Poisson's ratio
	double getPoissonsRatio() const {return nu;}
	//! Returns permittivity
	double getPermittivity() const {return epsr;}
	//! Returns the mesh dimensions (2 or 3)
	unsigned getDim() const {return dim;}
	//! Returns the maximum allowed displacement for a result
	double getMaxDisplacement() const {return maxDisplacement;}
	//! Returns the density
	double getDensity() const {return rho;}
	//! Returns the volume fraction
	double getVolFrac() const {return volfrac;}
	//! Returns the weight between compliance and capacitance
	double getWeight() const {return weight;}
	//! Returns the compliance scale factor 
	double getComplianceScale() const {return complianceScale;}
	//! Returns the capacitance scale factor 
	double getCapacitanceScale() const {return capacitanceScale;}
	//! Returns the Exodus mesh file name
	const std::string& getMeshFileName() const {return meshFilename;}
	//!Returns the mesh file format 
	Topologies::MeshFileFormat getMeshFileFormat() const {return inputMFF;};
	//! Returns the vector of boundary condition definitions for the structural problem
	const std::vector<BoundaryCondition>& getStructuralBCVec() const {return bcVecLinearElastic;} 
	//! Returns the vector of boundary condition definitions for the Laplace problem
	const std::vector<BoundaryCondition>& getLaplaceBCVec() const {return bcVecLaplace;} 
	//! Returns the vector of load condition definitions
  const std::vector<std::vector<LoadCondition<double>>>& getStructuralLCVec() const {return lcVVLinearElastic;}
	//! Returns the vector of Laplace non-homogeneous boundary conditions 
  const std::vector<std::vector<LoadCondition<double>>>& getLaplaceLCVec() const {return lcVVLaplace;}
private:
	static BCType parseBCType(const std::string& inBCStr);
	static CoordinateSystem::Type parseCSType(const std::string& inCSTStr);
	void setFileFormat(const pugi::xml_node& rootNode);
	BoundaryCondition parseGeoBC(const pugi::xml_node& rootNode);
	LoadCondition<double> parseGeoLC(const pugi::xml_node& rootNode);
	CoordinateSystem::Type readCoordinateSystemAttribute(const pugi::xml_node& rootNode) const;

	unsigned dim = 2;
	double rho = 1.0;
	double E = 1.0;
	double nu = 0.25;
	double epsr = 2.0;
	double maxDisplacement = 1e16;
	double volfrac = 0.5;
	double weight = 0.5;
	double capacitanceScale = 1.0;
	double complianceScale = 1.0;
	Topologies::MeshFileFormat inputMFF = Topologies::mffUnknown;
	std::string meshFilename;
	std::vector<BoundaryCondition> bcVecLinearElastic;
	std::vector<BoundaryCondition> bcVecLaplace;
	std::vector<std::vector<LoadCondition<double>>> lcVVLinearElastic;
	std::vector<std::vector<LoadCondition<double>>> lcVVLaplace;
};

#endif

