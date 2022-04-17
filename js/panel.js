import { GUI } from './three.js/examples/jsm/libs/lil-gui.module.min.js';

function refresh_scene (scene, meshOpe){
    scene.remove(scene.children[scene.children.length - 1]); 
    meshOpe.display(scene);
}

var SingletonPanel = {

    gui : new GUI(),
    b_show_edges : true,
    b_show_verts : true,
    point_width : 1,
    display : undefined,
    actions : undefined,
    init: function(scene, meshOpe) {
        this.display = this.gui.addFolder( 'display' );
        const p_disp = { 'show edges': true, 'show points': true, 'point width': 1. };
        this.display.add( p_disp, 'show edges' ).onChange( function ( val ) {
            SingletonPanel.b_show_edges = val;
            refresh_scene(scene, meshOpe);
        } );
        this.display.add( p_disp, 'show points' ).onChange( function ( val ) {
            SingletonPanel.b_show_verts = val;
            refresh_scene(scene, meshOpe);
        } );
        this.display.add( p_disp, 'point width', 0.1, 2, 0.1 ).onChange( function ( val ) {
            SingletonPanel.point_width = val;
            refresh_scene(scene, meshOpe);
        } );
        this.actions = this.gui.addFolder('actions');
        const p_actions = { 'Compute FrameField': function(){ meshOpe.computeFF(); }};//meshOpe.computeFF};//, 'Compute Param': true, 'Compute Quad mesh': 1. };
        this.actions.add( p_actions, 'Compute FrameField' );
    }
};

export{ SingletonPanel };