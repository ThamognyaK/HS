import DropDowns from './dropdown'
import ButtonU from './button'
import SideMenu from './sidemenu'
import LargeNavBar from './largeNavbar'
import { Show, Hide } from '@chakra-ui/react'

const NavBar = () => {
    const breakpoint = 'xl'
    return (
        <>
            <div className="font-druk fontplsdruk text-indigo bg-white py-2 ">
                <div className="drop-shadow-xl bg-white p-2">
                    <Show below={breakpoint}>
                        <SideMenu />
                    </Show>
                    <Hide below={breakpoint}>
                        <LargeNavBar />
                    </Hide>
                </div>
            </div>
        </>
    )
}

export default NavBar
