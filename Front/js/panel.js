import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';

function refresh_scene(scene, meshOpe) {
    scene.remove(scene.children[scene.children.length - 1]);
    meshOpe.display(scene);
}

class DisplayOptions
{
    bIsShowEdges = true;
    bIsShowVerts = true;
    pointWidth = 1;
}


class GuiPanel {
    constructor(scene) {
        this.gui = new GUI();
        this.displayOptions = new DisplayOptions()
        this.display = undefined;
        this.actions = undefined;
        this.scene = scene
    }

    init() {
        this.display = this.gui.addFolder('display');
        const p_disp = { 'show edges': true, 'show points': true, 'point width': 1. };
        this.display.add(p_disp, 'show edges').onChange(((val) => {
            this.displayOptions.bIsShowEdges = val;
            this.scene.mesh.updateDisplay(this.displayOptions)
        }).bind(this));
        this.display.add(p_disp, 'show points').onChange(((val) => {
            this.displayOptions.bIsShowVerts = val;
            this.scene.mesh.updateDisplay(this.displayOptions)
        }).bind(this));
        this.display.add(p_disp, 'point width', 0.1, 2, 0.1).onChange(((val)=> {
            this.displayOptions.pointWidth = val;
            this.scene.mesh.updateDisplay(this.displayOptions)
        }).bind(this));
        this.actions = this.gui.addFolder('actions');
        const p_actions = { 'Compute FrameField': (()=> { this.scene.mesh.computeFF(); }).bind(this) };
        this.actions.add(p_actions, 'Compute FrameField');
    }
};

export { GuiPanel };
