import {
    Menu,
    MenuButton,
    MenuList,
    MenuItem,
    MenuItemOption,
    MenuGroup,
    MenuOptionGroup,
    MenuDivider,
    useDisclosure,
    Button
} from '@chakra-ui/react'

import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons'

import Underline from './underline'

/*
 *
                    _expanded={{ bg: 'white' }}
                    _hover={{ bg: 'white' }}
                    _focus={{ bg: 'white' }}
 * */

function DropDowns(props) {
    const { isOpen, onOpen, onClose } = useDisclosure()
    return (
        <>
            <Menu isOpen={isOpen}>
                <MenuButton
                    as={Button}
                    px={4}
                    py={2}
                    bg="white"
                    transition="all 0.2s"
                    borderRadius="10"
                    borderWidth="0px"
                    boxShadow="none"
                    _hover={{ bg: 'gray.200' }}
                    _expanded={{ bg: 'gray.200' }}
                    _active={{ borderColor: 'gray.200' }}
                    onMouseEnter={onOpen}
                    onMouseLeave={onClose}
                    rightIcon={isOpen ? <ChevronUpIcon /> : <ChevronDownIcon />}
                >
                    {props.name}
                </MenuButton>
                <MenuList onMouseEnter={onOpen} onMouseLeave={onClose}>
                    {props.arr.map(function (val, i) {
                        return (
                            // eslint-disable-next-line react/jsx-key
                            <MenuItem>
                                <div className="text-black text-md">
                                    <Underline
                                        hre={val
                                            .replace(/\s+/g, '-')
                                            .toLowerCase()}
                                        name={val}
                                        color="black"
                                    />
                                </div>
                            </MenuItem>
                        )
                    })}
                </MenuList>
            </Menu>
        </>
    )
}

export default DropDowns
