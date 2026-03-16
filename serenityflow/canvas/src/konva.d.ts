/**
 * Minimal Konva type declarations for SerenityFlow canvas.
 * These cover the Konva API surface used in this project.
 * Not exhaustive — add types as needed during migration.
 */

declare namespace Konva {
    interface NodeConfig {
        x?: number;
        y?: number;
        width?: number;
        height?: number;
        visible?: boolean;
        listening?: boolean;
        draggable?: boolean;
        opacity?: number;
        id?: string;
        name?: string;
        [key: string]: any;
    }

    interface ContainerConfig extends NodeConfig {
        container?: string | HTMLDivElement;
        [key: string]: any;
    }

    class Node {
        x(): number;
        x(val: number): this;
        y(): number;
        y(val: number): this;
        width(): number;
        width(val: number): this;
        height(): number;
        height(val: number): this;
        visible(): boolean;
        visible(val: boolean): this;
        listening(): boolean;
        listening(val: boolean): this;
        opacity(): number;
        opacity(val: number): this;
        destroy(): void;
        remove(): void;
        getAbsolutePosition(): { x: number; y: number };
        setAbsolutePosition(pos: { x: number; y: number }): this;
        getAbsoluteTransform(): Transform;
        on(evtStr: string, handler: (e: KonvaEventObject<any>) => void): this;
        off(evtStr: string): this;
        getStage(): Stage | null;
        getLayer(): Layer | null;
        getParent(): Node | null;
        moveToTop(): this;
        moveToBottom(): this;
        getAttr(attr: string): any;
        setAttr(attr: string, val: any): this;
        setAttrs(attrs: Record<string, any>): this;
        position(): { x: number; y: number };
        position(pos: { x: number; y: number }): this;
        absolutePosition(): { x: number; y: number };
        absolutePosition(pos: { x: number; y: number }): this;
        scale(): { x: number; y: number };
        scale(val: { x: number; y: number }): this;
        scaleX(): number;
        scaleX(val: number): this;
        scaleY(): number;
        scaleY(val: number): this;
        id(): string;
        id(val: string): this;
        name(): string;
        name(val: string): this;
        show(): this;
        hide(): this;
        cache(): this;
        clearCache(): this;
        draw(): this;
        draggable(): boolean;
        draggable(val: boolean): this;
        parent: Container | null;
        getClientRect(config?: { skipTransform?: boolean }): { x: number; y: number; width: number; height: number };
        find(selector: string): Node[];
        findOne(selector: string): Node | undefined;
        hasName(name: string): boolean;
        addName(name: string): this;
        className: string;
    }

    class Container extends Node {
        add(...children: Node[]): this;
        removeChildren(): this;
        destroyChildren(): this;
        getChildren(): Node[];
        children: Node[];
        toDataURL(config?: any): string;
        clip(): { x: number; y: number; width: number; height: number };
        clip(val: { x: number; y: number; width: number; height: number }): this;
    }

    class Stage extends Container {
        constructor(config: ContainerConfig);
        container(): HTMLDivElement;
        getPointerPosition(): { x: number; y: number } | null;
        batchDraw(): this;
        getLayers(): Layer[];
        setPointersPositions(evt: any): void;
        toDataURL(config?: any): string;
        toCanvas(config?: any): HTMLCanvasElement;
    }

    class Layer extends Container {
        constructor(config?: NodeConfig);
        batchDraw(): this;
        canvas: { _canvas: HTMLCanvasElement };
        draw(): this;
        clear(): this;
    }

    class Group extends Container {
        constructor(config?: NodeConfig);
    }

    class Shape extends Node {
        constructor(config?: NodeConfig & {
            sceneFunc?: (context: any, shape: Shape) => void;
            fill?: string;
            stroke?: string;
            strokeWidth?: number;
            dash?: number[];
            hitStrokeWidth?: number;
            [key: string]: any;
        });
        fill(): string;
        fill(val: string): this;
        stroke(): string;
        stroke(val: string): this;
        strokeWidth(): number;
        strokeWidth(val: number): this;
        dash(): number[];
        dash(val: number[]): this;
        dashOffset(): number;
        dashOffset(val: number): this;
        shadowColor(): string;
        shadowColor(val: string): this;
        shadowBlur(): number;
        shadowBlur(val: number): this;
        shadowOffset(): { x: number; y: number };
        shadowOffset(val: { x: number; y: number }): this;
        shadowOpacity(): number;
        shadowOpacity(val: number): this;
        cornerRadius(): number;
        cornerRadius(val: number): this;
        hitFunc(): Function;
        hitFunc(val: Function): this;
    }

    class Animation {
        constructor(func: (frame: { time: number; timeDiff: number; lastTime: number; frameRate: number }) => void, layer?: Layer | null);
        start(): this;
        stop(): this;
        isRunning(): boolean;
    }

    class Rect extends Shape {
        constructor(config?: NodeConfig & {
            fill?: string;
            stroke?: string;
            strokeWidth?: number;
            cornerRadius?: number;
            dash?: number[];
            shadowColor?: string;
            shadowBlur?: number;
            shadowOffset?: { x: number; y: number };
            shadowOpacity?: number;
            shadowEnabled?: boolean;
            hitFunc?: Function;
            fillPatternImage?: HTMLImageElement;
            fillPatternRepeat?: string;
        });
        radius(): number;
        radius(val: number): this;
    }

    class Circle extends Shape {
        constructor(config?: NodeConfig & {
            radius?: number;
            fill?: string;
            stroke?: string;
            strokeWidth?: number;
        });
        radius(): number;
        radius(val: number): this;
    }

    class Line extends Shape {
        constructor(config?: NodeConfig & {
            points?: number[];
            stroke?: string;
            strokeWidth?: number;
            tension?: number;
            bezier?: boolean;
            dash?: number[];
            lineCap?: string;
            lineJoin?: string;
            globalCompositeOperation?: string;
        });
        points(): number[];
        points(val: number[]): this;
        tension(): number;
        tension(val: number): this;
        bezier(): boolean;
        bezier(val: boolean): this;
    }

    class Text extends Shape {
        constructor(config?: NodeConfig & {
            text?: string;
            fontSize?: number;
            fontFamily?: string;
            fontStyle?: string;
            fill?: string;
            align?: string;
            verticalAlign?: string;
            wrap?: string;
            ellipsis?: boolean;
            padding?: number;
            lineHeight?: number;
        });
        text(): string;
        text(val: string): this;
        fontSize(): number;
        fontSize(val: number): this;
        fontFamily(): string;
        fontFamily(val: string): this;
        getTextWidth(): number;
        measureSize(text: string): { width: number; height: number };
        align(): string;
        align(val: string): this;
    }

    class Image extends Shape {
        constructor(config?: NodeConfig & {
            image?: HTMLImageElement | HTMLCanvasElement;
            crop?: { x: number; y: number; width: number; height: number };
        });
        image(): HTMLImageElement | HTMLCanvasElement;
        image(val: HTMLImageElement | HTMLCanvasElement): this;
        crop(): { x: number; y: number; width: number; height: number };
        crop(val: { x: number; y: number; width: number; height: number }): this;
        static fromURL(url: string, callback: (img: Image) => void): void;
    }

    class Path extends Shape {
        constructor(config?: NodeConfig & {
            data?: string;
            fill?: string;
            stroke?: string;
            strokeWidth?: number;
        });
    }

    class Transformer extends Node {
        constructor(config?: NodeConfig & {
            nodes?: Node[];
            rotateEnabled?: boolean;
            enabledAnchors?: string[];
            boundBoxFunc?: (oldBox: any, newBox: any) => any;
        });
        nodes(): Node[];
        nodes(val: Node[]): this;
        getActiveAnchor(): string | null;
    }

    class Transform {
        copy(): Transform;
        invert(): Transform;
        point(pos: { x: number; y: number }): { x: number; y: number };
        translate(x: number, y: number): Transform;
        scale(sx: number, sy: number): Transform;
        getMatrix(): number[];
    }

    interface KonvaEventObject<E> {
        target: Node;
        currentTarget: Node;
        evt: E;
        cancelBubble: boolean;
        type: string;
    }
}

// Konva is loaded as a global <script> tag
declare var Konva: typeof Konva;
