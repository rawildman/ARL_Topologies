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

#include <Eigen/IterativeLinearSolvers>
//#include <Eigen/SparseLU>
#include "femproblem.h"
#include "element.h"
#include "lintri.h"
#include "linquad.h"
#include "lintetra.h"
#include "trilinhex.h"
#include "mesh2d.h"
#include "mesh3d.h"
#include "REP/tomesh.h"

FEMProblem::FEMProblem(const Topologies::TOMesh& inMesh, const Topologies::GenericMaterial& baseMat) :
	numFreeDOFs(0),
	invalid(false),
	itTol(1e-6)
{
	assert(baseMat.getNumParameters() > 2); // Assumes 3 material properties: density and 2 Lame parameters
	dim = inMesh.dimNum();
	if(dim == 2)
		probMesh = std::unique_ptr<FEMMesh>(new Mesh2D(inMesh, baseMat));
	else if(dim == 3)
		probMesh = std::unique_ptr<FEMMesh>(new Mesh3D(inMesh, baseMat));
	// Check element types (tris and tets need lower tolerance on iterative solver)
	if(checkForSimplex())
		itTol = 1e-12;
}

bool FEMProblem::checkForSimplex() const
{
	bool foundSimplex = false;
	for(std::size_t k = 0; k < probMesh->getNumElements() && !foundSimplex; ++k)
		foundSimplex |= probMesh->getNumElementNodes(k) == (dim + 1);
	return foundSimplex;
}

Point3D FEMProblem::getMeshPoint(std::size_t kn) const
{
	if(probMesh->getDim() == 2)
	{
		Point2D pt = *dynamic_cast<Mesh2D*>(probMesh.get())->getNode(kn);
		return Point3D(pt.x, pt.y, 0.);
	}
	return *dynamic_cast<Mesh3D*>(probMesh.get())->getNode(kn);
}
