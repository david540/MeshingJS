
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import {mergeVertices} from './GeometryUtils.js';
import * as interfaceUtils from './interfaceUtils.js';
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js';

class Mesh {
    constructor(path = './examples/emerald.obj') {
        this.filePath = path
        this.object = new THREE.Group()
        this.mesh = null
        this.wireframe = null
        this.pointsCloud = null
    }

    async loadMesh() {
        this.object = new THREE.Group()

        //Load object
        const loadingManager = new THREE.LoadingManager();
        const objLoader = new OBJLoader(loadingManager);
        await objLoader.load(this.filePath, ((obj) => {
            obj.position.fromArray([0, 0, 0]);
            obj.name = "mesh"
            this.mesh = obj.children[0]
            let newGeometry = mergeVertices(this.mesh.geometry, 1e-8)
            this.mesh.geometry = newGeometry
            this.object.add(this.mesh);

            //set wireframe
            this.wireframe = new THREE.LineSegments(new THREE.WireframeGeometry(this.mesh.geometry), new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 }));
            this.wireframe.name = "wireframe"
            this.object.add(this.wireframe)

            //set points cloud
            this.computePointsCloud()
        }).bind(this));

    }

    computePointsCloud() {
        const loader = new THREE.TextureLoader();
        const texture = loader.load( '../examples/disc.png' );

        const pointsMaterial = new THREE.PointsMaterial( {
            color: new THREE.Color( 255, 0, 0 ),
            map: texture,
            size: 1,
            alphaTest: 0.5

        } );

        this.pointsCloud = new THREE.Points(this.mesh.geometry, pointsMaterial)
        this.pointsCloud.name = "pointsCloud";
        this.object.add(this.pointsCloud);
    }


    updateDisplay(displayOptions) {
        this.pointsCloud.visible = displayOptions.bIsShowVerts
        this.wireframe.visible = displayOptions.bIsShowEdges
        this.pointsCloud.material.size = displayOptions.pointWidth
    }

    extractVertices()
    {
        if(!this.mesh)
        {
            return []
        }

        return interfaceUtils.CreateVectorDouble(this.mesh.geometry.getAttribute("position").array);
    }

    createVertex

    extractFaceIndices()
    {
        if(!this.mesh)
        {
            return []
        }

        return interfaceUtils.CreateVectorInt(this.mesh.geometry.index.array);
    }

    computeFF() {
        console.log("Mesh indexes : ")
        let data = interfaceUtils.ExtractArray(Module.meshMagic(this.extractFaceIndices(), this.extractVertices()))
        
        let directionsArray = new Float32Array(data.slice(0, data.length/2))
        let oppositeDirectionsArray = new Float32Array(data.slice(0, data.length/2).map(function(x) {return -x}))
        let verticesArray = new Float32Array(data.slice(data.length/2))
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute( 'position', new THREE.BufferAttribute( verticesArray, 3 ) );
        geometry.setAttribute( 'normal', new THREE.BufferAttribute( directionsArray, 3 ) );
        const material = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
        const mesh = new THREE.Mesh( geometry, material )

        const geometry2 = new THREE.BufferGeometry();
        geometry2.setAttribute( 'position', new THREE.BufferAttribute( verticesArray, 3 ) );
        geometry2.setAttribute( 'normal', new THREE.BufferAttribute( oppositeDirectionsArray, 3 ) );
        const material2 = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
        const mesh2 = new THREE.Mesh( geometry2, material2 )
        
        let helpers = new THREE.Group()
        let helper = new VertexNormalsHelper( mesh, 1, new THREE.Color(255, 0, 0), 3 );
        helper.name = "frameField"
        helpers.add(helper)
        let helper2 = new VertexNormalsHelper( mesh2, 1, new THREE.Color(255, 0, 0), 3 );
        helper2.name = "frameField"
        helpers.add(helper2)

        this.object.add(helpers)
    }
}

export { Mesh };
