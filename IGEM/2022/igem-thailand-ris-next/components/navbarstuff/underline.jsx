const Underline = props => {
    return (
        <>
            <a
                href={props.hre}
                className={`relative py-1 before:content-[''] before:absolute before:block before:w-full before:h-[3px] 
                before:bottom-0 before:left-0 before:bg-${props.color}
                before:hover:scale-x-100 before:scale-x-0 before:origin-top-left
                before:transition before:ease-in-out before:duration-300`}
            >
                {props.name}
            </a>
        </>
    )
}

export default Underline
