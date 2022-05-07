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

#include <vector>
#include <memory>
#include <fstream>
#include <string>

#include "structuralcap.h"
#include "linearelasticproblem.h"
#include "laplaceproblem.h"
#include "REP/topoptrep.h"
#include "REP/tomesh.h"
#include "IO/exotxtmeshloader.h"
#include "UTIL/helper.h"
#include "point2d.h"
#include "point3d.h"

using namespace Topologies;

namespace {
	const std::function<double(double,double)> identity = [](double val, double){return val;};
	const std::function<double(double,double)> permittivity = [](double mat, double opt) {
		return (mat - 1.0) * opt + 1.0;};
	const std::vector<MaterialFunction> optimizationParameters = {identity, std::multiplies<double>(), std::multiplies<double>(), permittivity};

	double totalObjectiveFunction(const double weight, const double complianceScale, 
			const double capacitanceScale, const FEMProblems& fems)
	{
		const std::pair<double, bool> compliance = fems.upLinearElasticProblem->computeCompliance(); 
		const std::pair<double, bool> capacitance = fems.upLaplaceProblem->computeCompliance();
		std::cout << "Comp: " << compliance.first << ", cap: " << capacitance.first << std::endl;
		const double weightedCompliance = weight * compliance.first / complianceScale;
		const double weightedCapacitance = (1.0 - weight) * capacitance.first / capacitanceScale;
		std::cout << "W Comp: " << weightedCompliance << ", w cap: " << weightedCapacitance << std::endl;
		return weightedCompliance - weightedCapacitance;
	}

	std::vector<double> totalGradient(const double weight, const double complianceScale, 
			const double capacitanceScale, const TopOptRep& inTOR, 
			const TOMesh& resMesh, const FEMProblems& fems)
	{
		auto gLinearElastic = HelperNS::vecScalarMult(weight / complianceScale, 
				inTOR.applyDiffRep(fems.upLinearElasticProblem->gradCompliance(resMesh)));
		auto gLaplace = HelperNS::vecScalarMult(-(1.0 - weight) / capacitanceScale, 
				inTOR.applyDiffRep(fems.upLaplaceProblem->gradCompliance(resMesh)));
		return HelperNS::vecSum(gLinearElastic, gLaplace);
	}
}

StructuralCap::StructuralCap(const std::string& inpFileName)
{
	FEMInputLoader inptParser;
	inptParser.parseFile(inpFileName);
	// Convert to Lame parameters
	double E = inptParser.getYoungsMod(), nu = inptParser.getPoissonsRatio();
	double lambda = E*nu/((1. + nu)*(1. - 2.*nu));
	double mu = 0.5*E/(1. + nu);
	const std::vector<double> matParams = {
		inptParser.getDensity(),
		lambda,
		mu,
		inptParser.getPermittivity()};
	baseMat = GenericMaterial(matParams);
	// Copy remaining input parameters
	maxDisplacement = inptParser.getMaxDisplacement();
	volfrac = inptParser.getVolFrac();
	dim = inptParser.getDim();
	weight = inptParser.getWeight();
	complianceScale = inptParser.getComplianceScale();
	capacitanceScale = inptParser.getCapacitanceScale();
	// Load boundary conditions
	bcVecLinearElastic = inptParser.getStructuralBCVec();
	bcVecLaplace = inptParser.getLaplaceBCVec();
	lcVVLinearElastic = inptParser.getStructuralLCVec();
	lcVVLaplace = inptParser.getLaplaceLCVec();
}

FEMProblems StructuralCap::setupAndSolveFEM(const TopOptRep& inTOR, std::size_t kLoad) const
{
	// Set up mesh and fem problem
	std::unique_ptr<TOMesh> resMesh = inTOR.getAnalysisMesh();
	assert(resMesh);
	// Linear elastic
	auto upLEProb = std::make_unique<LinearElasticProblem>(*resMesh, baseMat, optimizationParameters);
	// Solve problem and get compliance
	upLEProb->changeBoundaryConditionsTo(generateExoBCVecLinearElastic(resMesh.get(), kLoad));

	// Laplace
	auto upLaplaceProb = std::make_unique<LaplaceProblem>(*resMesh, baseMat, optimizationParameters);
	// Solve problem and get compliance
	upLaplaceProb->changeBoundaryConditionsTo(generateExoBCVecLaplace(resMesh.get(), kLoad));

	return {std::move(upLEProb), std::move(upLaplaceProb)};
}

std::pair<double, bool> StructuralCap::operator()(const TopOptRep& inTOR) const
{
	std::pair<double, bool> rval(0., true);
	for(std::size_t k = 0; k < lcVVLinearElastic.size() && rval.second; ++k)
	{
		// Loop over each load case and sum compliance
		auto FEMs = setupAndSolveFEM(inTOR, k);
		rval.first += totalObjectiveFunction(weight, complianceScale, capacitanceScale, FEMs);
		rval.second &= FEMs.upLinearElasticProblem->validRun() && FEMs.upLaplaceProblem->validRun();
	}
	return rval;
}

void StructuralCap::f(const TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const
{	
	std::pair<double, bool> res = (*this)(inTOR);
	std::vector<double> ofvVec(2);
	ofvVec[0] = res.first;
	ofvVec[1] = inTOR.computeVolumeFraction();
	bool valid = res.second;
	if(res.first > maxDisplacement)
		valid = false;
	outRes = std::make_pair(ofvVec, valid);
}

void StructuralCap::g(const TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const
{
	if(!inTOR.hasAnalyticalDerivative()) // No analytical derivative is available, use finite differences
		TopOptObjFun::g(inTOR, outRes);
	else
	{
		outRes.first = std::vector<double>(inTOR.getDataSize(), 0.);
		outRes.second = true;
		std::unique_ptr<TOMesh> resMesh = inTOR.getAnalysisMesh();
		assert(resMesh);
		for(std::size_t k = 0; k < lcVVLinearElastic.size() && outRes.second; ++k)
		{
			auto FEMs = setupAndSolveFEM(inTOR, k);
			outRes.first = HelperNS::vecSum(outRes.first, totalGradient(weight, complianceScale, 
						capacitanceScale, inTOR, *resMesh, FEMs));
			outRes.second &= FEMs.upLinearElasticProblem->validRun() && FEMs.upLaplaceProblem->validRun();
		}
	}
}

void StructuralCap::fAndG(const TopOptRep& inTOR, std::pair<std::vector<double>, bool>& fRes, 
												std::pair<std::vector<double>, bool>& gRes) const
{
	// First check that there's a valid representation derivative
	if(!inTOR.hasAnalyticalDerivative()) // No analytical derivative is available, use finite differences
	{
		f(inTOR, fRes);
		TopOptObjFun::g(inTOR, gRes);
	}
	else
	{
		bool valid = true;
		double fval = 0.;
		gRes.first = std::vector<double>(inTOR.getDataSize(), 0.);
		gRes.second = true;
		std::unique_ptr<TOMesh> resMesh = inTOR.getAnalysisMesh();
		assert(resMesh);
		for(std::size_t k = 0; k < lcVVLinearElastic.size() && valid; ++k)
		{
			// Loop over each load case and sum compliance
			auto FEMs = setupAndSolveFEM(inTOR, k);
			valid &= FEMs.upLinearElasticProblem->validRun() && FEMs.upLaplaceProblem->validRun();
			// f: objective function value
			fval += totalObjectiveFunction(weight, complianceScale, capacitanceScale, FEMs);
			// g: gradient of obj. fun 
			gRes.first = HelperNS::vecSum(gRes.first, totalGradient(weight, complianceScale, 
						capacitanceScale, inTOR, *resMesh, FEMs));
		}
		valid &= fval < maxDisplacement;
		// Finish f, add volume fraction as second goal
		fRes.first = {fval, inTOR.computeVolumeFraction()};
		// Add valid flags
		fRes.second = valid;
		gRes.second = valid;
	}
}

// Constraints
void StructuralCap::c(const TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const
{
	double val = inTOR.computeVolumeFraction() - volfrac;
	std::vector<double> ofvVec(1, val);
	outRes = std::make_pair(ofvVec, true);
}

void StructuralCap::gc(const TopOptRep& inTOR, std::pair<std::vector<double>, bool>& outRes) const
{
	std::vector<double> dVF = inTOR.computeGradVolumeFraction();
	if(dVF.empty())
		TopOptObjFun::gc(inTOR, outRes);
	else
		outRes = std::make_pair(std::move(dVF), true);
}

std::vector<ExoBC> StructuralCap::generateExoBCVec(const TOMesh* const inMesh, const std::vector<BoundaryCondition>& bcVec, const std::vector<LoadCondition<double>>& lcVec) const
{
	// First boundary conditions
	std::vector<ExoBC> outEBC;
	outEBC.reserve(bcVec.size() + lcVec.size());
	for(auto const& bc : bcVec)
	{
		outEBC.push_back(ExoBC(dim));
		ExoBC& curebc = outEBC.back();
		curebc.dim = dim;
		curebc.isSupport = true;
		std::vector<bool> supps = bc.getFixedCoords();
		curebc.xsup = supps[0];
		curebc.ysup = supps[1];
		if(dim == 3)
			curebc.zsup = supps[2];
		curebc.nodeIDVec = bc.applyBC(inMesh);
	}	
	// Now load conditions
	for(auto const& lc : lcVec)
	{
		outEBC.push_back(ExoBC(dim));
		ExoBC& curebc = outEBC.back();
		curebc.dim = dim;
		curebc.isSupport = false;
		lc.applyLC(inMesh, curebc.nodeIDVec);
		std::vector<double> loadVec = lc.getLoadVec();
		assert(loadVec.size() >= 2);
		curebc.loadVec = Point3D(loadVec[0], loadVec[1], loadVec.size() > 2 ? loadVec[2] : 0.);
		curebc.ct = lc.getCoordinateSystemType();
	}
	return outEBC;
}
std::vector<ExoBC> StructuralCap::generateExoBCVecLinearElastic(const TOMesh* const inMesh, std::size_t kLoad) const
{
	return generateExoBCVec(inMesh, bcVecLinearElastic, lcVVLinearElastic[kLoad]);
}

std::vector<ExoBC> StructuralCap::generateExoBCVecLaplace(const TOMesh* const inMesh, std::size_t kLoad) const
{
	return generateExoBCVec(inMesh, bcVecLaplace, lcVVLaplace[kLoad]);
}

// Input parsing

void FEMInputLoader::parse(const pugi::xml_document& xmldoc)
{
	parse(xmldoc.child("fem"));
}	

void FEMInputLoader::parse(const pugi::xml_node& rootNode)
{
	using namespace InputLoader;
	try
	{
		// Read required inputs
		E = readDoublePCData(rootNode, "youngs_modulus");
		nu = readDoublePCData(rootNode, "poissons_ratio");
		epsr = readDoublePCData(rootNode, "relative_permittivity");
		volfrac = readDoublePCData(rootNode, "volume_fraction");
		dim = readUnsignedPCData(rootNode, "dimension");
		weight = readDoublePCData(rootNode, "objective_weight");
		complianceScale = readDoublePCData(rootNode, "compliance_scale");
		capacitanceScale = readDoublePCData(rootNode, "capacitance_scale");
		if(dim != 2 && dim != 3)
			throw(ParseException(petInvalidNumericalInput, "dimension must be 2 or 3"));
		if(weight < 0.0 || weight > 1.0)
			throw(ParseException(petInvalidNumericalInput, "objective_weight must be in [0, 1]"));
	}
	catch(ParseException pe)
	{
		errorMessage(pe, true);
	}
	// Optional inputs
	try{maxDisplacement = readDoublePCData(rootNode, "max_displacement");}
	catch(ParseException pe){maxDisplacement = 1e16;}
	try{rho = readDoublePCData(rootNode, "density");}
	catch(ParseException pe){rho = 1.;}
	try{meshFilename = readAndCheckFileNamePCData(rootNode, "mesh_file");}
	catch(ParseException pe){}//Only necessary for file_bc input
	try{setFileFormat(rootNode);}
	catch(ParseException pe){}// Only necessary for file_bc input
	if(!meshFilename.empty() && inputMFF == mffExodus)
	{
		// Check mesh dimensions
		unsigned probDim = ExoTxtMeshLoader::readMeshDimension(meshFilename);
		if(probDim != dim)
		{
			ParseException pe(petInvalidNumericalInput, "Wrong number of dimensions in exodus mesh file!");
			errorMessage(pe, true);
		}
	}
	// GeoBC input
	for(pugi::xml_node cn = rootNode.child("geo_structural_bc"); cn; cn = cn.next_sibling("geo_structural_bc"))
	  bcVecLinearElastic.push_back(parseGeoBC(cn));
	for(pugi::xml_node cn = rootNode.child("geo_laplace_bc"); cn; cn = cn.next_sibling("geo_laplace_bc"))
		bcVecLaplace.push_back(parseGeoBC(cn));
	
	// GeoLC input
	lcVVLinearElastic.push_back(std::vector<LoadCondition<double>>());
	for(pugi::xml_node cn = rootNode.child("geo_structural_lc"); cn; cn = cn.next_sibling("geo_structural_lc"))	
		lcVVLinearElastic.back().push_back(parseGeoLC(cn));
	
	lcVVLaplace.push_back(std::vector<LoadCondition<double>>());
	for(pugi::xml_node cn = rootNode.child("geo_laplace_lc"); cn; cn = cn.next_sibling("geo_laplace_lc"))	
		lcVVLaplace.back().push_back(parseGeoLC(cn));	
}

CoordinateSystem::Type FEMInputLoader::parseCSType(const std::string& inCSTStr)
{
	using namespace InputLoader;
	if(inCSTStr == "cartesian")
		return CoordinateSystem::Type::cartesian;
	else if(inCSTStr == "cylindrical")
		return CoordinateSystem::Type::cylindrical;
	else if(inCSTStr == "spherical")
		return CoordinateSystem::Type::spherical;
	throw ParseException(petUnknownInput, "Unknown coordinate system type");
}

BCType FEMInputLoader::parseBCType(const std::string& inBCStr)
{
	if(inBCStr == "v_line")
		return bctVLine;
	else if(inBCStr == "h_line")
		return bctHLine;
	else if(inBCStr == "line_segment_2")
		return bctLineSeg;
	else if(inBCStr == "point_2")
		return bctPoint2D;
	else if(inBCStr == "polygon_2")
		return bctPolygon2D;
	else if(inBCStr == "xy_plane")
		return bctXYPlane;
	else if(inBCStr == "yz_plane")
		return bctYZPlane;
	else if(inBCStr == "xz_plane")
		return bctXZPlane;
	else if(inBCStr == "point_3")
		return bctPoint3D;
	else if(inBCStr == "line_segment_3")
		return bctLineSeg3D;
	return bctUnknown;
}

BoundaryCondition FEMInputLoader::parseGeoBC(const pugi::xml_node& rootNode)
{
	using namespace InputLoader;
	BCType outBCT;
	bool xsup, ysup, zsup = false;
	std::unique_ptr<GeometricEntity> upGE;
	try
	{
		outBCT = parseBCType(rootNode.child("geometry").first_child().name());
		upGE = GeometryEntityFactory::createGeometricEntity(outBCT, rootNode.child("geometry").first_child());
		xsup = readBoolPCData(rootNode, "x_support");
		ysup = readBoolPCData(rootNode, "y_support");
		if(dim == 3)
			zsup = readBoolPCData(rootNode, "z_support");
	}
	catch(ParseException pe)
	{
		errorMessage(pe, true);
	}
	return BoundaryCondition(outBCT, xsup, ysup, zsup, std::move(upGE));
}

CoordinateSystem::Type FEMInputLoader::readCoordinateSystemAttribute(const pugi::xml_node& rootNode) const
{
	using namespace InputLoader;
	CoordinateSystem::Type ct;
	std::string cs;
	try
	{
		cs = readStringAttribute(rootNode, "coordinate_type");
		ct = parseCSType(cs);
	}
	catch(ParseException pe)
	{
		ct = CoordinateSystem::Type::cartesian;
		if(!cs.empty())
			std::cerr << "Warning: Unknown coordinate system in load_vector: " << cs << "\nDefaulting to cartesian" << std::endl;
	}
	return ct;
}

LoadCondition<double> FEMInputLoader::parseGeoLC(const pugi::xml_node& rootNode)
{
	using namespace InputLoader;
	BCType outBCT;
	std::vector<double> vecRes;
	std::unique_ptr<GeometricEntity> upGE;
  try
  {
		outBCT = parseBCType(rootNode.child("geometry").first_child().name());
    upGE = GeometryEntityFactory::createGeometricEntity(outBCT, rootNode.child("geometry").first_child());
		vecRes = readDoubleVecPCData(rootNode, "load_vector");
    if(vecRes.size() != dim)
      throw ParseException(petInvalidNumericalInput, "Wrong number of dimensions in load_vector!");
  }
  catch(ParseException pe)
  {
    errorMessage(pe, true);
  }
	CoordinateSystem::Type ct = readCoordinateSystemAttribute(rootNode.child("load_vector"));
  return LoadCondition<double>(outBCT, vecRes, std::move(upGE), ct);
}

void FEMInputLoader::setFileFormat(const pugi::xml_node& rootNode)
{
	using namespace InputLoader;
	std::string retval = readStringPCData(rootNode, "file_format");
	inputMFF = parseMeshFileFormat(retval);
	if(inputMFF != mffExodus && inputMFF != mffGMSH)
		throw ParseException(petUnknownInput, std::move(retval));
}

