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
import { on } from 'events'

/*
 *
                    _expanded={{ bg: 'white' }}
                    _hover={{ bg: 'white' }}
                    _focus={{ bg: 'white' }}
 * */

function DropDownU(props) {
    const { isOpen, onOpen, onClose } = useDisclosure()
    return (
        <>
            <div className="text-start">
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
                        onClick={isOpen ? onClose : onOpen}
                        rightIcon={
                            isOpen ? <ChevronUpIcon /> : <ChevronDownIcon />
                        }
                    >
                        <div className="text-sm">
                            <div
                                className={
                                    isOpen ? 'text-indigo' : 'text-black'
                                }
                            >
                                {props.name}
                            </div>
                        </div>
                    </MenuButton>
                    <MenuList onClick={isOpen ? onClose : onOpen}>
                        {props.arr.map(function (val, i) {
                            return (
                                // eslint-disable-next-line react/jsx-key
                                <MenuItem>
                                    <div className="text-black text-sm text-start focus:text-indigo">
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
            </div>
        </>
    )
}

export default DropDownU
